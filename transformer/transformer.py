# đường đi của mô hình transformer
import tensorflow as tf
from transformer.encoder.encoder import EncoderLayer
from transformer.decoder.decoder import DecoderLayer
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super().__init__()

        self.encoder = EncoderLayer(num_layers, d_model, num_heads, dff, input_vocab_size, dropout_rate)
        self.decoder = DecoderLayer(num_layers, d_model, num_heads, dff, target_vocab_size, dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size, activation='softmax')

    def call(self, inputs, targets, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        # Pass the inputs through the encoder
        enc_output = self.encoder(inputs, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # Pass the encoder output, inputs and look ahead mask to the decoder
        dec_output, attention_weights = self.decoder(targets, enc_output, look_ahead_mask, dec_padding_mask)

        # Pass the decoder output through the final layer
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights
