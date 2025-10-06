import os
import wandb
import argparse
import numpy as np
import yaml
import time
import json
import pdb
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

"""
IMPORT YOUR MODEL HERE
"""
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.training.train_eval_loop import load_model, load_ema_model
from vint_train.training.train_utils import evaluate, evaluate_nomad
from vint_train.training.logger import Logger
from vint_train.visualizing.action_utils import visualize_traj_pred
from vint_train.visualizing.distance_utils import visualize_dist_pred
from vint_train.visualizing.visualize_utils import to_numpy

def setup_device_and_config(config):
    """Setup CUDA device and configuration."""
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif type(config["gpu_ids"]) == int:
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in config["gpu_ids"]]
        )
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    cudnn.benchmark = True
    return device

def load_trained_model(config, device):
    """Load a trained model from checkpoint."""
    # Create the model architecture
    if config["model_type"] == "nomad":
        if config["vision_encoder"] == "nomad_vint":
            vision_encoder = NoMaD_ViNT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        else: 
            raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")
            
        noise_pred_net = ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=config["encoding_size"],
                down_dims=config["down_dims"],
                cond_predict_scale=config["cond_predict_scale"],
            )
        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
        
        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )

        noise_scheduler = DDPMScheduler(
            num_train_timesteps=config["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
    else:
        raise ValueError(f"Model {config['model_type']} not supported")

    # Load the checkpoint
    checkpoint_path = config["checkpoint_path"]
    print(f"Loading model from {checkpoint_path}")
    
    if config["model_type"] == "nomad":
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        
        # Also load EMA model if available
        ema_model = None
        ema_checkpoint_path = checkpoint_path.replace(".pth", "_ema.pth")
        if os.path.exists(ema_checkpoint_path):
            print(f"Loading EMA model from {ema_checkpoint_path}")
            from diffusers.training_utils import EMAModel
            ema_model = EMAModel(model=model, power=0.75)
            ema_checkpoint = torch.load(ema_checkpoint_path, map_location=device)
            ema_model.load_state_dict(ema_checkpoint)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        load_model(model, config["model_type"], checkpoint)
        ema_model = None
        noise_scheduler = None

    # Multi-GPU support
    if len(config["gpu_ids"]) > 1:
        model = nn.DataParallel(model, device_ids=config["gpu_ids"])
    model = model.to(device)
    
    if ema_model is not None:
        ema_model = ema_model.to(device)

    return model, ema_model, noise_scheduler

def load_evaluation_datasets(config):
    """Load datasets for evaluation."""
    test_dataloaders = {}
    
    # Set up transforms
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Set default values
    if "context_type" not in config:
        config["context_type"] = "temporal"
    if "clip_goals" not in config:
        config["clip_goals"] = False

    # Load datasets
    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        
        # Set defaults
        if "negative_mining" not in data_config:
            data_config["negative_mining"] = True
        if "goals_per_obs" not in data_config:
            data_config["goals_per_obs"] = 1
        if "end_slack" not in data_config:
            data_config["end_slack"] = 0
        if "waypoint_spacing" not in data_config:
            data_config["waypoint_spacing"] = 1

        # Load test dataset
        if "test" in data_config:
            dataset = ViNT_Dataset(
                data_folder=data_config["data_folder"],
                data_split_folder=data_config["test"],
                dataset_name=dataset_name,
                image_size=config["image_size"],
                waypoint_spacing=data_config["waypoint_spacing"],
                min_dist_cat=config["distance"]["min_dist_cat"],
                max_dist_cat=config["distance"]["max_dist_cat"],
                min_action_distance=config["action"]["min_dist_cat"],
                max_action_distance=config["action"]["max_dist_cat"],
                negative_mining=data_config["negative_mining"],
                len_traj_pred=config["len_traj_pred"],
                learn_angle=config.get("learn_angle", True),
                context_size=config["context_size"],
                context_type=config["context_type"],
                end_slack=data_config["end_slack"],
                goals_per_obs=data_config["goals_per_obs"],
                normalize=config["normalize"],
                goal_type=config["goal_type"],
            )
            
            dataset_type = f"{dataset_name}_test"
            test_dataloaders[dataset_type] = DataLoader(
                dataset,
                batch_size=config.get("eval_batch_size", config["batch_size"]),
                shuffle=False,  # Keep deterministic for evaluation
                num_workers=config.get("num_workers", 0),
                drop_last=False,
            )

    return test_dataloaders, transform

def evaluate_model_on_dataset(
    model, 
    ema_model, 
    noise_scheduler, 
    dataloader, 
    transform, 
    device, 
    config, 
    dataset_name,
    output_dir
):
    """Evaluate model on a single dataset."""
    print(f"Evaluating on {dataset_name}...")
    
    if config["model_type"] == "nomad" and ema_model is not None:
        # Use EMA model for NoMaD evaluation
        return evaluate_nomad(
            eval_type=dataset_name,
            ema_model=ema_model,
            dataloader=dataloader,
            transform=transform,
            device=device,
            noise_scheduler=noise_scheduler,
            goal_mask_prob=config.get("goal_mask_prob", 0.5),
            project_folder=output_dir,
            epoch=0,
            print_log_freq=config.get("print_log_freq", 100),
            wandb_log_freq=config.get("wandb_log_freq", 10),
            image_log_freq=config.get("image_log_freq", 1000),
            num_images_log=config.get("num_images_log", 8),
            eval_fraction=1.0,  # Evaluate on full dataset
            use_wandb=config.get("use_wandb", False),
        )
    else:
        # Use regular model for ViNT/GNM evaluation
        return evaluate(
            eval_type=dataset_name,
            model=model,
            dataloader=dataloader,
            transform=transform,
            device=device,
            project_folder=output_dir,
            normalized=config["normalize"],
            epoch=0,
            alpha=config.get("alpha", 0.5),
            learn_angle=config.get("learn_angle", True),
            num_images_log=config.get("num_images_log", 8),
            use_wandb=config.get("use_wandb", False),
            eval_fraction=1.0,  # Evaluate on full dataset
            use_tqdm=True,
        )

def save_evaluation_results(results, output_dir):
    """Save evaluation results to JSON file."""
    results_file = os.path.join(output_dir, "evaluation_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")

def main(config):
    """Main evaluation function."""
    # Setup device and configuration
    device = setup_device_and_config(config)
    
    # Create output directory
    output_dir = config.get("output_dir", "evaluation_results")
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    output_dir = os.path.join(output_dir, f"eval_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config used for evaluation
    config_save_path = os.path.join(output_dir, "eval_config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(config, f)
    
    # Setup wandb if enabled
    if config.get("use_wandb", False):
        wandb.login()
        wandb.init(
            project=config.get("project_name", "navigation_eval"),
            name=f"eval_{timestamp}",
            config=config,
        )
    
    # Load trained model
    model, ema_model, noise_scheduler = load_trained_model(config, device)
    
    # Load evaluation datasets
    test_dataloaders, transform = load_evaluation_datasets(config)
    
    if not test_dataloaders:
        print("No test datasets found. Please check your configuration.")
        return
    
    # Evaluate on each dataset
    all_results = {}
    
    for dataset_name, dataloader in test_dataloaders.items():
        try:
            if config["model_type"] == "nomad":
                # For NoMaD, evaluation function doesn't return metrics directly
                # We'll capture them from the logger or wandb
                evaluate_model_on_dataset(
                    model, ema_model, noise_scheduler, dataloader, 
                    transform, device, config, dataset_name, output_dir
                )
                # For now, just record that evaluation was completed
                all_results[dataset_name] = {"status": "completed"}
            else:
                # For ViNT/GNM models, we get metrics returned
                dist_loss, action_loss, total_loss = evaluate_model_on_dataset(
                    model, ema_model, noise_scheduler, dataloader,
                    transform, device, config, dataset_name, output_dir
                )
                
                all_results[dataset_name] = {
                    "distance_loss": float(dist_loss),
                    "action_loss": float(action_loss),
                    "total_loss": float(total_loss),
                    "num_samples": len(dataloader.dataset)
                }
                
                print(f"{dataset_name} Results:")
                print(f"  Distance Loss: {dist_loss:.4f}")
                print(f"  Action Loss: {action_loss:.4f}")
                print(f"  Total Loss: {total_loss:.4f}")
                print(f"  Samples: {len(dataloader.dataset)}")
                print()
                
        except Exception as e:
            print(f"Error evaluating {dataset_name}: {str(e)}")
            all_results[dataset_name] = {"error": str(e)}
    
    # Save results
    save_evaluation_results(all_results, output_dir)
    
    # Log summary to wandb
    if config.get("use_wandb", False):
        wandb.log({"evaluation_summary": all_results})
        wandb.finish()
    
    print("Evaluation completed!")
    print(f"Results saved in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained navigation model")
    
    # Required arguments
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to the trained model checkpoint"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to the model config file used during training"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=None,
        help="Batch size for evaluation (defaults to training batch size)"
    )
    parser.add_argument(
        "--use_wandb", 
        action="store_true",
        help="Enable wandb logging"
    )
    parser.add_argument(
        "--eval_fraction", 
        type=float, 
        default=1.0,
        help="Fraction of dataset to evaluate (for quick testing)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    with open("config/defaults.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)
    
    config.update(user_config)
    
    # Override with command line arguments
    config["checkpoint_path"] = args.checkpoint
    config["output_dir"] = args.output_dir
    if args.eval_batch_size is not None:
        config["eval_batch_size"] = args.eval_batch_size
    if args.use_wandb:
        config["use_wandb"] = True
    config["eval_fraction"] = args.eval_fraction
    
    # Disable training mode
    config["train"] = False
    
    print("Evaluation Configuration:")
    print(f"  Checkpoint: {config['checkpoint_path']}")
    print(f"  Model Type: {config['model_type']}")
    print(f"  Output Directory: {config['output_dir']}")
    print(f"  Use WandB: {config.get('use_wandb', False)}")
    print(f"  Eval Fraction: {config['eval_fraction']}")
    print()
    
    main(config)