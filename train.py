# Train mô hình
import tensorflow as tf
from transformer.transformer import Transformer
from transformer.preprocessing.text_processor import TextProcessor

# Load preprocessed data
en_data_path = "data/preprocessed/en_data.txt"
vi_data_path = "data/preprocessed/vi_data.txt"
text_processor = TextProcessor(en_data_path, vi_data_path)

# Create the transformer model
transformer = Transformer(
    num_layers=4,
    d_model=128,
    num_heads=8,
    dff=512,
    input_vocab_size=text_processor.en_vocab_size + 2,
    target_vocab_size=text_processor.vi_vocab_size + 2,
    pe_input=text_processor.en_vocab_size + 2,
    pe_target=text_processor.vi_vocab_size + 2,
)

# Restore the latest checkpoint
checkpoint_path = tf.train.latest_checkpoint("checkpoints")
transformer.load_weights(checkpoint_path)

# Translate input sentence
input_sentence = "Đây là một ví dụ về việc dịch máy sử dụng Transformer."
input_sequence = text_processor.preprocess_vi_text(input_sentence)
output_sequence = transformer.translate(input_sequence)
output_sentence = text_processor.postprocess_en_text(output_sequence)

# Print the translation
print("Input sentence: ", input_sentence)
print("Output sentence: ", output_sentence)
