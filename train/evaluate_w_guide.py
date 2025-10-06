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
import matplotlib.pyplot as plt

# Handle diffusers import compatibility
try:
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
except ImportError as e:
    print(f"Warning: Could not import DDPMScheduler: {e}")
    print("Please update your environment or use pip install diffusers==0.21.4")
    exit(1)

"""
IMPORT YOUR MODEL HERE
"""
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.training.train_eval_loop import load_model, load_ema_model
from vint_train.training.train_utils import evaluate, evaluate_nomad, get_action, normalize_data, unnormalize_data
from vint_train.training.logger import Logger
from vint_train.visualizing.action_utils import visualize_traj_pred, plot_trajs_and_points
from vint_train.visualizing.distance_utils import visualize_dist_pred
from vint_train.visualizing.visualize_utils import to_numpy, from_numpy
from vint_train.data.data_utils import VISUALIZATION_IMAGE_SIZE

# Import guidance modules
try:
    import sys
    sys.path.append('../deployment/src')
    from guide import PathGuide, PathOpt
    GUIDANCE_AVAILABLE = True
    print("Using guidance modules from deployment directory")
except ImportError:
    print("Warning: Guidance modules not available. Please run copy_guidance_modules.sh first.")
    GUIDANCE_AVAILABLE = False

# ACTION STATS (should match training)
ACTION_STATS = {}
ACTION_STATS['min'] = np.array([-2.5, -4])
ACTION_STATS['max'] = np.array([5, 4])

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
        raise ValueError(f"Model {config['model_type']} not supported for guided evaluation")

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
                # batch_size=config.get("eval_batch_size", config["batch_size"]),
                batch_size=config['eval_batch_size'],
                shuffle=False,  # Keep deterministic for evaluation
                num_workers=config.get("num_workers", 0),
                drop_last=False,
            )

    return test_dataloaders, transform

def model_output_guided(
    model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    num_samples: int,
    device: torch.device,
    use_guidance: bool = True,
    pathguide = None,
    goal_positions: torch.Tensor = None,
    scale_factor: float = 1.0,
):
    """
    Model output with optional gradient-based guidance during diffusion.
    Based on the guidance logic from navigate.py.
    """
    # Setup goal masking (no mask for goal-conditioned, full mask for unconditional)
    no_mask = torch.zeros((batch_goal_images.shape[0],)).long().to(device)
    obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=no_mask)
    obsgoal_cond = obsgoal_cond.repeat_interleave(num_samples, dim=0)

    # Initialize action from Gaussian noise
    noisy_diffusion_output = torch.randn(
        (len(obsgoal_cond), pred_horizon, action_dim), device=device)
    diffusion_output = noisy_diffusion_output

    # Diffusion process with optional guidance
    for k in noise_scheduler.timesteps[:]:
        with torch.no_grad():
            # Predict noise
            noise_pred = model(
                "noise_pred_net",
                sample=diffusion_output,
                timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
                global_cond=obsgoal_cond
            )

            # Inverse diffusion step (remove noise)
            diffusion_output = noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=diffusion_output
            ).prev_sample

        # Apply guidance if enabled (based on navigate.py logic)
        if use_guidance and pathguide is not None and goal_positions is not None:
            interval1 = 6
            period = 1
            if k <= interval1:
                if k % period == 0:
                    if k > 2:
                        # Apply gradient guidance for collision avoidance and goal reaching
                        # Temporarily enable gradients for guidance computation
                        with torch.enable_grad():
                            diffusion_output.requires_grad_(True)
                            grad, cost_list = pathguide.get_gradient(
                                diffusion_output, 
                                goal_pos=goal_positions, 
                                scale_factor=scale_factor
                            )
                        grad_scale = 1.0
                        diffusion_output = diffusion_output.detach() - grad_scale * grad
                    else:
                        if k >= 0 and k <= 2:
                            # Multiple gradient steps for early timesteps
                            diffusion_output_tmp = diffusion_output.detach().clone()
                            for i in range(1):
                                with torch.enable_grad():
                                    diffusion_output_tmp.requires_grad_(True)
                                    grad, cost_list = pathguide.get_gradient(
                                        diffusion_output_tmp, 
                                        goal_pos=goal_positions, 
                                        scale_factor=scale_factor
                                    )
                                diffusion_output_tmp = diffusion_output_tmp.detach() - grad
                            diffusion_output = diffusion_output_tmp

    # Convert to actions
    gc_actions = get_action(diffusion_output, ACTION_STATS)
    
    # Also compute distance prediction
    obsgoal_cond_flat = obsgoal_cond[:batch_goal_images.shape[0]].flatten(start_dim=1)
    gc_distance = model("dist_pred_net", obsgoal_cond=obsgoal_cond_flat)

    return {
        'gc_actions': gc_actions,
        'gc_distance': gc_distance,
    }

def model_output_standard(
    model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    num_samples: int,
    device: torch.device,
):
    """
    Standard model output without guidance (for comparison).
    """
    from vint_train.training.train_utils import model_output
    
    return model_output(
        model,
        noise_scheduler,
        batch_obs_images,
        batch_goal_images,
        pred_horizon,
        action_dim,
        num_samples,
        device,
    )

def evaluate_with_guidance(
    model, 
    ema_model, 
    noise_scheduler, 
    dataloader, 
    transform, 
    device, 
    config, 
    dataset_name,
    output_dir,
    use_guidance=True
):
    """Evaluate model with optional guidance."""
    print(f"Evaluating on {dataset_name} {'with' if use_guidance else 'without'} guidance...")
    
    # Initialize guidance if available and requested
    pathguide = None
    if use_guidance and GUIDANCE_AVAILABLE:
        pathguide = PathGuide(device, ACTION_STATS)
        print("PathGuide initialized for gradient-based guidance")
    elif use_guidance and not GUIDANCE_AVAILABLE:
        print("Warning: Guidance requested but not available. Running without guidance.")
        use_guidance = False
    
    # Use EMA model for evaluation if available
    eval_model = ema_model.averaged_model if ema_model is not None else model
    eval_model.eval()
    
    # Metrics storage
    metrics = defaultdict(list)
    
    # Create visualization directory
    viz_dir = os.path.join(output_dir, "visualize", dataset_name, "guided" if use_guidance else "standard")
    os.makedirs(viz_dir, exist_ok=True)
    
    num_batches = len(dataloader)
    scale_factor = 4.0  # As used in navigate.py
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dataloader, desc=f"Evaluating {dataset_name}")):
            (
                obs_image,
                goal_image,
                action_label,
                dist_label,
                goal_pos,
                dataset_index,
                action_mask,
            ) = data

            # Prepare observation images
            obs_images = torch.split(obs_image, 3, dim=1)
            viz_obs_image = obs_images[-1]  # Last frame for visualization
            obs_images = [transform(obs_image).to(device) for obs_image in obs_images]
            obs_image = torch.cat(obs_images, dim=1)

            # Prepare goal images
            viz_goal_image = goal_image
            goal_image = transform(goal_image).to(device)
            
            # Move labels to device
            dist_label = dist_label.to(device)
            action_label = action_label.to(device)
            action_mask = action_mask.to(device)
            goal_pos = goal_pos.to(device)

            pred_horizon = action_label.shape[1]
            action_dim = action_label.shape[2]
            
            # Generate cost map for guidance if enabled
            if use_guidance and pathguide is not None:
                # Use first observation image for cost map generation
                from PIL import Image as PILImage
                import torchvision.transforms.functional as TF
                
                # Convert tensor back to PIL for pathguide
                obs_pil = TF.to_pil_image(viz_obs_image[0])
                pathguide.get_cost_map_via_tsdf(obs_pil)
            
            # Get model predictions
            if use_guidance and pathguide is not None:
                model_output_dict = model_output_guided(
                    eval_model,
                    noise_scheduler,
                    obs_image,
                    goal_image,
                    pred_horizon,
                    action_dim,
                    num_samples=config.get("num_samples", 1),
                    device=device,
                    use_guidance=True,
                    pathguide=pathguide,
                    goal_positions=goal_pos[:1],  # Use first goal position
                    scale_factor=scale_factor,
                )
            else:
                model_output_dict = model_output_standard(
                    eval_model,
                    noise_scheduler,
                    obs_image,
                    goal_image,
                    pred_horizon,
                    action_dim,
                    num_samples=config.get("num_samples", 1),
                    device=device,
                )
            
            dist_pred = model_output_dict['gc_distance']
            action_pred = model_output_dict['gc_actions']
            
            # Handle multiple samples - take mean of distance predictions
            if dist_pred.shape[0] != dist_label.shape[0]:
                # Distance prediction has been expanded for multiple samples
                batch_size = dist_label.shape[0]
                num_samples = dist_pred.shape[0] // batch_size
                dist_pred = dist_pred.view(batch_size, num_samples, -1).mean(dim=1)
            
            # Handle multiple samples - take first sample for action prediction
            if action_pred.shape[0] != action_label.shape[0]:
                batch_size = action_label.shape[0]
                num_samples = action_pred.shape[0] // batch_size
                seq_len = action_pred.shape[1]
                action_dim = action_pred.shape[2]
                action_pred = action_pred.view(batch_size, num_samples, seq_len, action_dim)[:, 0]  # Take first sample
            
            # Compute metrics
            # Distance metrics
            dist_mse = torch.nn.functional.mse_loss(dist_pred.squeeze(-1), dist_label.float())
            dist_mae = torch.nn.functional.l1_loss(dist_pred.squeeze(-1), dist_label.float())
            metrics['distance_mse'].append(dist_mse.item())
            metrics['distance_mae'].append(dist_mae.item())
            
            # Action metrics (only where mask is valid)
            valid_mask = action_mask.bool()
            if valid_mask.any():
                action_mse = torch.nn.functional.mse_loss(
                    action_pred[valid_mask], 
                    action_label[valid_mask]
                )
                metrics['action_mse'].append(action_mse.item())
                
                # Endpoint error
                final_pred_pos = action_pred[valid_mask][:, -1, :2]
                final_true_pos = action_label[valid_mask][:, -1, :2]
                endpoint_error = torch.norm(final_pred_pos - final_true_pos, dim=-1).mean()
                metrics['endpoint_error'].append(endpoint_error.item())
                
                # Cosine similarity
                pos_cos_sim = torch.nn.functional.cosine_similarity(
                    action_pred[valid_mask][:, :, :2].flatten(start_dim=1),
                    action_label[valid_mask][:, :, :2].flatten(start_dim=1),
                    dim=-1
                ).mean()
                metrics['position_cosine_sim'].append(pos_cos_sim.item())
            
            # Save some visualizations (first few batches)
            if batch_idx < config.get("num_images_log", 8):
                save_trajectory_comparison(
                    viz_obs_image[0],
                    viz_goal_image[0], 
                    action_pred[0],
                    action_label[0],
                    goal_pos[0],
                    batch_idx,
                    viz_dir,
                    guided=use_guidance
                )
    
    # Aggregate metrics
    aggregated_metrics = {}
    for key, values in metrics.items():
        if values:
            aggregated_metrics[f"{key}_mean"] = np.mean(values)
            aggregated_metrics[f"{key}_std"] = np.std(values)
            aggregated_metrics[f"{key}_median"] = np.median(values)
    
    return aggregated_metrics

def save_trajectory_comparison(obs_image, goal_image, pred_action, true_action, goal_pos, idx, save_dir, guided=False):
    """Save trajectory comparison visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Observation image
    obs_img = to_numpy(obs_image).transpose(1, 2, 0)
    axes[0].imshow(obs_img)
    axes[0].set_title('Observation')
    axes[0].axis('off')
    
    # Goal image
    goal_img = to_numpy(goal_image).transpose(1, 2, 0)
    axes[1].imshow(goal_img)
    axes[1].set_title('Goal')
    axes[1].axis('off')
    
    # Trajectory comparison
    pred_traj = to_numpy(pred_action)[:, :2]
    true_traj = to_numpy(true_action)[:, :2]
    goal_position = to_numpy(goal_pos)[:2]
    
    axes[2].plot(true_traj[:, 0], true_traj[:, 1], 'g-o', label='Ground Truth', markersize=4)
    axes[2].plot(pred_traj[:, 0], pred_traj[:, 1], 'r-o', label=f'Predicted {"(Guided)" if guided else "(Standard)"}', markersize=4)
    axes[2].plot(0, 0, 'bo', markersize=8, label='Start')
    axes[2].plot(goal_position[0], goal_position[1], 'mo', markersize=8, label='Goal')
    axes[2].set_title(f'Trajectories {"with Guidance" if guided else "Standard"}')
    axes[2].legend()
    axes[2].grid(True)
    axes[2].axis('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'trajectory_comparison_{idx}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def compare_guidance_methods(
    model, ema_model, noise_scheduler, dataloader, transform, device, config, dataset_name, output_dir
):
    """Compare guided vs non-guided evaluation."""
    print(f"Comparing guidance methods on {dataset_name}...")
    
    # Evaluate without guidance
    print("Evaluating without guidance...")
    standard_metrics = evaluate_with_guidance(
        model, ema_model, noise_scheduler, dataloader, transform, 
        device, config, f"{dataset_name}_standard", output_dir, use_guidance=False
    )
    
    # Evaluate with guidance
    print("Evaluating with guidance...")
    guided_metrics = evaluate_with_guidance(
        model, ema_model, noise_scheduler, dataloader, transform, 
        device, config, f"{dataset_name}_guided", output_dir, use_guidance=True
    )
    
    # Compute improvement metrics
    improvement_metrics = {}
    for key in standard_metrics:
        if key in guided_metrics:
            if 'mse' in key or 'mae' in key or 'error' in key:
                # Lower is better
                improvement = ((standard_metrics[key] - guided_metrics[key]) / standard_metrics[key]) * 100
            else:
                # Higher is better (cosine similarity)
                improvement = ((guided_metrics[key] - standard_metrics[key]) / standard_metrics[key]) * 100
            improvement_metrics[f"{key}_improvement_percent"] = improvement
    
    return {
        'standard': standard_metrics,
        'guided': guided_metrics,
        'improvement': improvement_metrics
    }

def save_evaluation_results(results, output_dir):
    """Save evaluation results to JSON file."""
    results_file = os.path.join(output_dir, "guided_evaluation_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")

def main(config):
    """Main evaluation function."""
    # Setup device and configuration
    device = setup_device_and_config(config)
    
    # Create output directory
    output_dir = config.get("output_dir", "guided_evaluation_results")
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    output_dir = os.path.join(output_dir, f"guided_eval_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config used for evaluation
    config_save_path = os.path.join(output_dir, "guided_eval_config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(config, f)
    
    # Setup wandb if enabled
    if config.get("use_wandb", False):
        wandb.login()
        wandb.init(
            project=config.get("project_name", "navigation_guided_eval"),
            name=f"guided_eval_{timestamp}",
            config=config,
        )
    
    # Check if guidance is available
    if not GUIDANCE_AVAILABLE:
        print("Warning: Guidance modules not available. Evaluation will compare standard model only.")
    
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
        if config.get("compare_guidance", True) and GUIDANCE_AVAILABLE:
            # Compare guided vs non-guided
            comparison_results = compare_guidance_methods(
                model, ema_model, noise_scheduler, dataloader,
                transform, device, config, dataset_name, output_dir
            )
            all_results[dataset_name] = comparison_results
            
            # Print comparison summary
            print(f"\n{dataset_name} Comparison Results:")
            print("=" * 50)
            for metric_type in ['standard', 'guided']:
                print(f"\n{metric_type.upper()} Results:")
                metrics = comparison_results[metric_type]
                for key, value in metrics.items():
                    if 'mean' in key:
                        print(f"  {key}: {value:.4f}")
            
            print("\nIMPROVEMENT Results:")
            for key, value in comparison_results['improvement'].items():
                print(f"  {key}: {value:.2f}%")
                
        else:
            # Evaluate with guidance only (or standard only if guidance not available)
            use_guidance = GUIDANCE_AVAILABLE and config.get("use_guidance", True)
            metrics = evaluate_with_guidance(
                model, ema_model, noise_scheduler, dataloader,
                transform, device, config, dataset_name, output_dir, use_guidance=use_guidance
            )
            all_results[dataset_name] = metrics
            
            print(f"\n{dataset_name} Results:")
            print("=" * 30)
            for key, value in metrics.items():
                if 'mean' in key:
                    print(f"  {key}: {value:.4f}")
    
    # Save results
    save_evaluation_results(all_results, output_dir)
    
    # Log summary to wandb
    if config.get("use_wandb", False):
        wandb.log({"guided_evaluation_summary": all_results})
        wandb.finish()
    
    print("\nGuided evaluation completed!")
    print(f"Results saved in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained navigation model with gradient-based guidance")
    
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
        default="guided_evaluation_results",
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
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=8,
        help="Number of action samples for diffusion"
    )
    parser.add_argument(
        "--compare_guidance", 
        action="store_true",
        help="Compare guided vs non-guided evaluation (default: just run with guidance)"
    )
    parser.add_argument(
        "--use_guidance", 
        action="store_true",
        default=True,
        help="Use gradient-based guidance (only used if compare_guidance is False)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    # with open("train/config/defaults.yaml", "r") as f:
    #     config = yaml.safe_load(f)
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # config.update(user_config)
    
    # Override with command line arguments
    config["checkpoint_path"] = args.checkpoint
    config["output_dir"] = args.output_dir
    if args.eval_batch_size is not None:
        config["eval_batch_size"] = args.eval_batch_size
    if args.use_wandb:
        config["use_wandb"] = True
    config["eval_fraction"] = args.eval_fraction
    config["num_samples"] = args.num_samples
    config["compare_guidance"] = args.compare_guidance
    config["use_guidance"] = args.use_guidance
    
    # Disable training mode
    config["train"] = False
    
    print("Guided Evaluation Configuration:")
    print(f"  Checkpoint: {config['checkpoint_path']}")
    print(f"  Model Type: {config['model_type']}")
    print(f"  Output Directory: {config['output_dir']}")
    print(f"  Use WandB: {config.get('use_wandb', False)}")
    print(f"  Eval Fraction: {config['eval_fraction']}")
    print(f"  Num Samples: {config['num_samples']}")
    print(f"  Compare Guidance: {config['compare_guidance']}")
    print(f"  Use Guidance: {config['use_guidance']}")
    print(f"  Guidance Available: {GUIDANCE_AVAILABLE}")
    print()
    
    main(config)