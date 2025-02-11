import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model


def encoder(input_dim, hidden_dim):
    encoder_input = Input(shape=(None, input_dim))
    encoder_lstm = LSTM(hidden_dim, return_state=True)
    _, state_h, state_c = encoder_lstm(encoder_input)  # Changed: we only need states
    encoder_states = [state_h, state_c]
    return encoder_input, encoder_states

def decoder(hidden_dim, output_dim):
    decoder_input = Input(shape=(None, output_dim))
    decoder_lstm = LSTM(hidden_dim, return_sequences=True, return_state=True)
    decoder_dense = Dense(output_dim, activation='softmax')  # Changed: renamed for clarity
    return decoder_input, decoder_lstm, decoder_dense

input_dim = 100 
hidden_dim = 256 
output_dim = 100 

def build_seq2seq(input_dim, hidden_dim, output_dim):
    # Get encoder parts
    encoder_input, encoder_states = encoder(input_dim, hidden_dim)
    
    # Get decoder parts
    decoder_input, decoder_lstm, decoder_dense = decoder(hidden_dim, output_dim)
    
    # Connect encoder and decoder
    decoder_outputs, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Create model
    model = Model([encoder_input, decoder_input], decoder_outputs)
    return model

# Build and compile model
model = build_seq2seq(input_dim, hidden_dim, output_dim)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.summary()
