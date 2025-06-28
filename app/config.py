# Configuration for LIBRAS to Portuguese application

# Model hyperparameters (must match training configuration)
MODEL_CONFIG = {
    "n_points": 21,          # Number of hand points
    "emb_size": 128,         # Embedding size
    "hidden_size": 256,      # LSTM hidden layer size
    "num_layers": 1,         # Number of LSTM layers
    "dropout": 0.5           # Dropout rate
}

# MediaPipe configuration
MEDIAPIPE_CONFIG = {
    "static_image_mode": False,
    "max_num_hands": 2,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5
}

# Interface configuration
UI_CONFIG = {
    "max_frames_video": 100,
    "camera_fps": 30,
    "sequence_length": 30,    # Maximum number of frames for temporal sequence
    "min_sequence_length": 5  # Minimum number of frames to make prediction
}

# Available models (in priority order)
MODEL_FILES = [
    "best_model.pt",
    "best_model_270625.pt", 
    "sign2text_model.pt"
]

# Tokenizer configuration
TOKENIZER_CONFIG = {
    "model_name": "neuralmind/bert-base-portuguese-cased",
    "add_special_tokens": False
}
