import tensorflow as tf

def build_transformer_model(input_shape, num_classes):
    """
    Cria um modelo Transformer simples para classificação de sequências de landmarks.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Masking()(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model