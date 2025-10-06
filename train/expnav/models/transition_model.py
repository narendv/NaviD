import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from vint_train.models.vint.self_attention import PositionalEncoding
from vint_train.models.nomad.nomad_vint import replace_bn_with_gn, NoMaD_ViNT


class TransitionModel(nn.Module):
    """
    Transition model that takes an encoded context vector and action to predict
    the resulting context vector after taking that action.
    
    This model learns the dynamics of how the encoded state changes based on actions.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        action_dim: int = 2,
        num_attention_heads: int = 8,
        num_attention_layers: int = 4,
        ff_dim_factor: int = 4,
        dropout: float = 0.1,
        max_sequence_length: int = 10,
    ) -> None:
        """
        Initialize the transition model.
        
        Args:
            d_model: Size of the context vector from NoMaD encoder
            action_dim: Dimension of action space (e.g., 2 for [v, w])
            action_encoding_size: Size to encode actions to
            num_attention_heads: Number of attention heads in transformer
            num_attention_layers: Number of transformer encoder layers
            ff_dim_factor: Factor for feedforward dimension
            dropout: Dropout probability
            max_sequence_length: Maximum sequence length for positional encoding
        """
        super().__init__()
        
        self.d_model = d_model
        self.action_dim = action_dim
        
        # Action encoder - maps raw actions to embedding space
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Calculate the full sequence dimension (context + action)
        self.sequence_dim = d_model
        
        # Positional encoding for transformer
        self.positional_encoding = PositionalEncoding(
            d_model=self.sequence_dim,
            max_seq_len=max_sequence_length,
            dropout=dropout
        )
        
        print(f"Creating PositionalEncoding with d_model={self.sequence_dim}")  # Debug print
        
        # Transformer encoder layers (similar to nomad_vint.py)
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.sequence_dim,
            nhead=num_attention_heads,
            dim_feedforward=ff_dim_factor * self.sequence_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        
        self.sa_encoder = nn.TransformerEncoder(
            self.sa_layer,
            num_layers=num_attention_layers
        )
        
        # Output projection to predict next context vector
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Optional residual connection parameter
        self.use_residual = True
        
    def forward(
        self, 
        context: torch.Tensor, 
        action: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the transition model.
        
        Args:
            context: Current context encoding from NoMaD [B, T, d_model]
            action: Action taken [batch_size, action_dim] or [B, action_dim]
            mask: Optional attention mask
            
        Returns:
            next_context: Predicted next context encoding [batch_size, d_model]
        """
        
        # Encode actions
        action_encoded = self.action_encoder(action).unsqueeze(1)  # [batch_size, 1, action_encoding_size]
        
        # Combine context and action into sequence
        sequence = torch.cat([context, action_encoded], dim=1)  # [batch_size, 1+seq_len, d_model]

        # Apply positional encoding
        sequence = self.positional_encoding(sequence)
        
        # Apply transformer encoder
        transformed_sequence = self.sa_encoder(sequence, src_key_padding_mask=mask)
        
        # Extract the final representation
        next_context = transformed_sequence[:, :-1, :]  # [batch_size, d_model]
        
        # Apply output projection
        next_context = self.output_projection(next_context)
        
        # Optional residual connection
        if self.use_residual:
            next_context = next_context + context
        
        return next_context
    
    def forward_sequence(
        self,
        context: torch.Tensor,
        action: torch.Tensor,
        return_intermediate: bool = False
    ) -> torch.Tensor:
        """
        Forward pass for a sequence of actions, applying them one by one.
        
        Args:
            context_vector: Initial context encoding [batch_size, d_model]
            action_sequence: Sequence of actions [batch_size, seq_len, action_dim]
            return_intermediate: If True, return all intermediate context vectors
            
        Returns:
            final_context: Final context vector after all actions [batch_size, d_model]
            or if return_intermediate=True:
            all_contexts: All context vectors [batch_size, seq_len+1, d_model]
        """
        _, seq_len, _ = action_sequence.shape
        current_context = context

        if return_intermediate:
            all_contexts = [current_context]
        
        # Apply actions sequentially
        for t in range(seq_len):
            action_t = action_sequence[:, t, :]  # [batch_size, action_dim]
            current_context = self.forward(current_context, action_t)
            
            if return_intermediate:
                all_contexts.append(current_context)
        
        if return_intermediate:
            return torch.stack(all_contexts, dim=1)  # [batch_size, seq_len+1, d_model]
        else:
            return current_context
    
    def predict_multi_step(
        self,
        context_vector: torch.Tensor,
        action_sequence: torch.Tensor,
        prediction_horizon: int = 1
    ) -> torch.Tensor:
        """
        Predict multiple steps into the future.
        
        Args:
            context_vector: Current context encoding [batch_size, d_model]
            action_sequence: Sequence of actions [batch_size, seq_len, action_dim]
            prediction_horizon: Number of steps to predict
            
        Returns:
            predicted_contexts: Predicted context vectors [batch_size, prediction_horizon, d_model]
        """
        predictions = []
        current_context = context_vector
        
        for step in range(prediction_horizon):
            if step < action_sequence.shape[1]:
                # Use provided action
                action = action_sequence[:, step, :]
            else:
                # Use last action (could be modified to use a learned policy)
                action = action_sequence[:, -1, :]
            
            current_context = self.forward(current_context, action)
            predictions.append(current_context)
        
        return torch.stack(predictions, dim=1)  # [batch_size, prediction_horizon, d_model]


class TransitionModelWithUncertainty(TransitionModel):
    """
    Extended transition model that also predicts uncertainty in the transition.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Additional head for uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, self.d_model),
            nn.Softplus()  # Ensure positive uncertainty values
        )
    
    def forward(
        self, 
        context_vector: torch.Tensor, 
        action: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_uncertainty: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional uncertainty estimation.
        
        Returns:
            next_context_vector: Predicted next context encoding
            uncertainty: Optional uncertainty estimate (if return_uncertainty=True)
        """
        # Get the base prediction
        next_context = super().forward(context_vector, action, mask)
        
        if return_uncertainty:
            # Predict uncertainty based on the transformed representation
            # We need to recompute to get intermediate representations
            batch_size = context_vector.shape[0]
            device = context_vector.device
            
            if action.dim() == 2:
                action = action.unsqueeze(1)
            
            action_encoded = self.action_encoder(action)
            action_projected = self.action_projection(action_encoded)
            context_projected = self.context_projection(context_vector).unsqueeze(1)
            sequence = torch.cat([context_projected, action_projected], dim=1)
            sequence = self.positional_encoding(sequence)
            transformed_sequence = self.transformer_encoder(sequence, src_key_padding_mask=mask)
            
            # Use the final representation for uncertainty prediction
            uncertainty = self.uncertainty_head(transformed_sequence[:, -1, :])
            
            return next_context, uncertainty
        else:
            return next_context, None


def create_transition_model(
    d_model: int = 256,
    action_dim: int = 2,
    model_type: str = "basic"
) -> nn.Module:
    """
    Factory function to create transition models.
    
    Args:
        d_model: Size of context vectors from NoMaD
        action_dim: Dimension of action space
        model_type: Type of model ("basic" or "uncertainty")
    
    Returns:
        Transition model instance
    """
    if model_type == "basic":
        return TransitionModel(
            d_model=d_model,
            action_dim=action_dim
        )
    elif model_type == "uncertainty":
        return TransitionModelWithUncertainty(
            d_model=d_model,
            action_dim=action_dim
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Example usage and testing
if __name__ == "__main__":
    # Test the transition model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create model
    model = TransitionModel(
        d_model=256,
        action_dim=2,
        num_attention_heads=8,
        num_attention_layers=4
    ).to(device)
    
    # Test data
    batch_size = 16
    context_vector = torch.randn(batch_size, 5, 256).to(device)
    action = torch.randn(batch_size, 2).to(device)  # [v, w] actions
    action_sequence = torch.randn(batch_size, 5, 2).to(device)  # Sequence of 5 actions
    
    print("Testing single action transition:")
    next_context = model(context_vector, action)
    print(f"Input context shape: {context_vector.shape}")
    print(f"Action shape: {action.shape}")
    print(f"Output context shape: {next_context.shape}")
    
    # print("\nTesting action sequence:")
    # final_context = model.forward_sequence(context_vector, action_sequence)
    # print(f"Action sequence shape: {action_sequence.shape}")
    # print(f"Final context shape: {final_context.shape}")
    
    # print("\nTesting with intermediate outputs:")
    # all_contexts = model.forward_sequence(context_vector, action_sequence, return_intermediate=True)
    # print(f"All contexts shape: {all_contexts.shape}")
    
    # print("\nTesting uncertainty model:")
    # uncertainty_model = TransitionModelWithUncertainty(
    #     d_model=256,
    #     action_dim=2
    # ).to(device)
    
    # next_context, uncertainty = uncertainty_model(context_vector, action, return_uncertainty=True)
    # print(f"Next context shape: {next_context.shape}")
    # print(f"Uncertainty shape: {uncertainty.shape}")
    
    # print("\nAll tests passed!")