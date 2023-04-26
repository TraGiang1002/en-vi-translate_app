import tensorflow as tf
from transformer.transformer import Transformer
from transformer.preprocessing.text_processor import TextProcessor

# load the preprocessed data
en_data_path = 'data/preprocessed/en_data.txt'
vi_data_path = 'data/preprocessed/vi_data.txt'
en_data = open(en_data_path, 'r', encoding='utf-8').read().split('\n')
vi_data = open(vi_data_path, 'r', encoding='utf-8').read().split('\n')

# create text processors for English and Vietnamese
en_processor = TextProcessor(en_data)
vi_processor = TextProcessor(vi_data)

# initialize the transformer model
transformer = Transformer(num_layers=4, d_model=128, num_heads=8, dff=512, input_vocab_size=en_processor.vocab_size,
                          target_vocab_size=vi_processor.vocab_size, pe_input=en_processor.vocab_size,
                          pe_target=vi_processor.vocab_size)

# load the trained model weights
checkpoint_path = "checkpoints/train"
ckpt = tf.train.Checkpoint(transformer=transformer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

# translate the input sentence
input_text = "xin chào"
input_seq = vi_processor.encode_sentence(input_text)
output_seq = transformer.evaluate(input_seq)
output_text = en_processor.decode_sentence(output_seq)

print(f"Input: {input_text}")
print(f"Output: {output_text}")
