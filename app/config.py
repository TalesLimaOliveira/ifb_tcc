# Configurações Otimizadas - LIBRAS Translator v2.0

# Parâmetros do modelo (devem coincidir com o treinamento)
MODEL_CONFIG = {
    "n_points": 21,          
    "emb_size": 128,         
    "hidden_size": 256,      
    "num_layers": 1,         
    "dropout": 0.5           
}

# Configuração MediaPipe otimizada
MEDIAPIPE_CONFIG = {
    "static_image_mode": False,
    "max_num_hands": 2,
    "min_detection_confidence": 0.7,    
    "min_tracking_confidence": 0.7,     
    "model_complexity": 0                
}

# Configuração de interface otimizada
UI_CONFIG = {
    "max_frames_video": 60,        
    "camera_fps": 15,              
    "sequence_length": 20,         
    "min_sequence_length": 3,      
    "frame_skip": 2,               
    "batch_size": 1                
}

# Modelos disponíveis (ordem de prioridade)
MODEL_FILES = [
    "best_model.pt",
    "model_270625.pt",
    "best_model_270625.pt", 
    "sign2text_model.pt"
]

# Configuração do tokenizer
TOKENIZER_CONFIG = {
    "model_name": "neuralmind/bert-base-portuguese-cased",
    "add_special_tokens": False
}
