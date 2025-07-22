"""
Modelos de Rede Neural para Tradução de LIBRAS

Componentes:
- CNNEncoder: Encoder convolucional para landmarks
- Sign2TextModel: Modelo CNN + RNN + FC completo
- ModelManager: Gerenciador para carregar e executar modelos
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer
import os


class CNNEncoder(nn.Module):
    """
    Convolutional Neural Network encoder for hand landmark features.
    
    This encoder processes 3D hand landmark coordinates through 1D convolutions
    to extract meaningful features for sequence modeling.
    
    Args:
        n_points (int): Number of landmark points per hand (default: 21)
        emb_size (int): Size of the output embedding dimension
    """
    
    def __init__(self, n_points, emb_size):
        super().__init__()
        self.n_points = n_points
        self.emb_size = emb_size
        
        # 1D convolution layers for feature extraction
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 64, 1)
        
        # Fully connected layer to produce final embedding
        self.fc = nn.Linear(64 * n_points, emb_size)

    def forward(self, x):
        """
        Forward pass through the CNN encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, n_points, 3)
                             or (batch, n_points, 3)
        
        Returns:
            torch.Tensor: Encoded features of shape (batch, seq_len, emb_size)
        """
        # Ensure input has sequence dimension
        if x.ndim == 3:
            x = x.unsqueeze(1)
            
        batch, seq_len, n_points, coords = x.shape
        
        # Reshape for convolution: (batch * seq_len, coords, n_points)
        x = x.view(-1, n_points, coords).permute(0, 2, 1)
        
        # Apply convolution layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        
        # Flatten and apply fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        # Reshape back to sequence format
        x = x.view(batch, seq_len, -1)
        
        return x


class Sign2TextModel(nn.Module):
    """
    Complete model for translating LIBRAS signs to Portuguese text.
    
    Architecture:
    1. CNN Encoder: Processes hand landmarks
    2. LSTM: Models temporal sequences
    3. Fully Connected: Maps to vocabulary space
    
    Args:
        n_points (int): Number of landmark points per hand
        emb_size (int): CNN encoder embedding size
        hidden_size (int): LSTM hidden state size
        vocab_size (int): Size of the output vocabulary
        num_layers (int): Number of LSTM layers
        dropout (float): Dropout rate for regularization
    """
    
    def __init__(self, n_points, emb_size, hidden_size, vocab_size, 
                 num_layers=1, dropout=0.5):
        super().__init__()
        
        # Store model configuration
        self.n_points = n_points
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Model components
        self.encoder = CNNEncoder(n_points, emb_size)
        self.rnn = nn.LSTM(
            emb_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        """
        Forward pass through the complete model.
        
        Args:
            x (torch.Tensor): Input landmarks of shape (batch, seq_len, n_points, 3)
        
        Returns:
            torch.Tensor: Logits for vocabulary tokens of shape (batch, seq_len, vocab_size)
        """
        # Encode landmarks to features
        encoded = self.encoder(x)
        
        # Process sequence with LSTM
        lstm_out, _ = self.rnn(encoded)
        
        # Map to vocabulary space
        output = self.fc(lstm_out)
        
        return output


class ModelManager:
    """
    Manager class for loading and handling the neural network model.
    
    This class handles model initialization, loading pre-trained weights,
    and provides a clean interface for model operations.
    """
    
    def __init__(self, model_config, tokenizer_model, device):
        """
        Initialize the model manager.
        
        Args:
            model_config (dict): Model hyperparameters
            tokenizer_model (str): Name of the tokenizer model
            device (torch.device): Device to run the model on
        """
        self.model_config = model_config
        self.tokenizer_model = tokenizer_model
        self.device = device
        self.model = None
        self.tokenizer = None
    
    def load_tokenizer(self):
        """
        Load the BERT tokenizer for Portuguese.
        
        Returns:
            BertTokenizer: Loaded tokenizer instance
        """
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_model)
        return self.tokenizer
    
    def initialize_model(self, vocab_size):
        """
        Initialize the neural network model.
        
        Args:
            vocab_size (int): Size of the vocabulary from tokenizer
        
        Returns:
            Sign2TextModel: Initialized model instance
        """
        self.model = Sign2TextModel(
            n_points=self.model_config["n_points"],
            emb_size=self.model_config["emb_size"],
            hidden_size=self.model_config["hidden_size"],
            vocab_size=vocab_size,
            num_layers=self.model_config["num_layers"],
            dropout=self.model_config["dropout"]
        )
        return self.model
    
    def load_pretrained_weights(self, model_paths=None):
        """
        Load pre-trained weights from available model files.
        
        Args:
            model_paths (list, optional): List of paths to try loading from.
                                        If None, uses default paths from config.
        
        Returns:
            tuple: (success: bool, loaded_path: str or None)
        """
        if self.model is None:
            raise ValueError("Model must be initialized before loading weights")
        
        # Use default paths if none provided
        if model_paths is None:
            from config import MODEL_FILES
            app_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(os.path.dirname(app_dir), 'models')
            model_paths = [os.path.join(models_dir, filename) for filename in MODEL_FILES]
        
        for path in model_paths:
            if os.path.exists(path):
                try:
                    # Load state dict with proper device mapping
                    state_dict = torch.load(path, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                    
                    # Move model to device and set to evaluation mode
                    self.model = self.model.to(self.device)
                    self.model.eval()
                    
                    return True, path
                except Exception as e:
                    # Log error but continue trying other paths
                    print(f"Failed to load {path}: {str(e)}")
                    continue
        
        return False, None
    
    def predict(self, landmarks_input):
        """
        Make prediction using the loaded model.
        
        Args:
            landmarks_input: Can be either:
                - torch.Tensor of shape (seq_len, n_points, 3) or (seq_len, max_hands, n_points, 3)
                - List of numpy arrays from MediaPipe processing
        
        Returns:
            str: Predicted text or error message
        """
        if self.model is None or self.tokenizer is None:
            return "Model or tokenizer not loaded"
        
        try:
            self.model.eval()
            with torch.no_grad():
                # Convert input to tensor if needed
                landmarks_tensor = self._prepare_landmarks_tensor(landmarks_input)
                
                if landmarks_tensor is None:
                    return "Invalid landmarks data"
                
                # Move to device
                landmarks_tensor = landmarks_tensor.to(self.device)
                
                # Forward pass
                outputs = self.model(landmarks_tensor)
                
                # Get predictions - use the last timestep or aggregate
                if outputs.dim() == 3:  # (batch, seq_len, vocab_size)
                    # Use mean pooling across sequence length for better results
                    outputs = outputs.mean(dim=1)  # (batch, vocab_size)
                
                predicted_ids = torch.argmax(outputs, dim=-1).cpu().numpy()
                
                # Handle batch dimension
                if predicted_ids.ndim > 1:
                    predicted_ids = predicted_ids[0]  # Take first batch
                
                # Convert to list if single value
                if predicted_ids.ndim == 0:
                    predicted_ids = [int(predicted_ids)]
                else:
                    predicted_ids = predicted_ids.tolist()
                
                # Decode using tokenizer
                predicted_text = self.tokenizer.decode(
                    predicted_ids, 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                # Clean up output
                predicted_text = predicted_text.strip()
                return predicted_text if predicted_text else "Nenhuma tradução disponível"
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Erro na predição: {str(e)}"
    
    def _prepare_landmarks_tensor(self, landmarks_input):
        """
        Convert landmarks input to the expected tensor format.
        
        Args:
            landmarks_input: Various formats of landmarks data
        
        Returns:
            torch.Tensor: Tensor of shape (1, seq_len, n_points, 3) or None if invalid
        """
        try:
            # If already a tensor
            if torch.is_tensor(landmarks_input):
                landmarks_tensor = landmarks_input
            else:
                # Convert list/numpy to tensor
                if isinstance(landmarks_input, list):
                    if len(landmarks_input) == 0:
                        return None
                    
                    # Handle list of numpy arrays (typical MediaPipe output)
                    landmarks_array = np.array(landmarks_input)
                    
                    # Expected shape after conversion: (seq_len, max_hands, n_points, 3)
                    if landmarks_array.ndim == 4:  # (seq_len, max_hands, n_points, 3)
                        # Combine both hands by taking mean or concatenating
                        if landmarks_array.shape[1] == 2:  # 2 hands
                            # Take the mean of both hands for simplicity
                            landmarks_array = np.mean(landmarks_array, axis=1)  # (seq_len, n_points, 3)
                        elif landmarks_array.shape[1] == 1:  # 1 hand
                            landmarks_array = landmarks_array.squeeze(1)  # (seq_len, n_points, 3)
                    
                    landmarks_tensor = torch.from_numpy(landmarks_array).float()
                else:
                    # Try to convert numpy array directly
                    landmarks_array = np.array(landmarks_input)
                    landmarks_tensor = torch.from_numpy(landmarks_array).float()
            
            # Ensure correct dimensions
            if landmarks_tensor.ndim == 2:  # (n_points, 3) - single frame
                landmarks_tensor = landmarks_tensor.unsqueeze(0)  # (1, n_points, 3)
            
            if landmarks_tensor.ndim == 3:  # (seq_len, n_points, 3)
                landmarks_tensor = landmarks_tensor.unsqueeze(0)  # (1, seq_len, n_points, 3)
            
            # Validate final shape
            expected_shape = (1, -1, self.model_config["n_points"], 3)  # -1 for variable seq_len
            if landmarks_tensor.shape[0] != 1 or landmarks_tensor.shape[2] != expected_shape[2] or landmarks_tensor.shape[3] != 3:
                print(f"Warning: Unexpected tensor shape {landmarks_tensor.shape}, expected {expected_shape}")
                
                # Try to fix common shape issues
                if landmarks_tensor.shape[2] != self.model_config["n_points"]:
                    # If we have more points, take the first n_points
                    if landmarks_tensor.shape[2] > self.model_config["n_points"]:
                        landmarks_tensor = landmarks_tensor[:, :, :self.model_config["n_points"], :]
                    else:
                        # If we have fewer points, pad with zeros
                        padding = torch.zeros(landmarks_tensor.shape[0], landmarks_tensor.shape[1], 
                                            self.model_config["n_points"] - landmarks_tensor.shape[2], 3)
                        landmarks_tensor = torch.cat([landmarks_tensor, padding], dim=2)
            
            return landmarks_tensor
            
        except Exception as e:
            print(f"Error preparing landmarks tensor: {str(e)}")
            return None
    
    def get_model_info(self):
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information including parameters and configuration
        """
        if self.model is None:
            return {"status": "No model loaded"}
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "status": "Model loaded",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "config": self.model_config,
            "device": str(self.device),
            "vocab_size": self.tokenizer.vocab_size if self.tokenizer else "Unknown"
        }
