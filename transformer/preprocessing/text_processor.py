# File xử lý dữ liệu văn bản trong quá trình huấn luyện mô hình
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import unicodedata
import re

class TextProcessor:
    def __init__(self, en_data_path, vi_data_path):
        # Load and preprocess the data
        self.en_tokenizer, self.en_vocab_size = self._create_tokenizer(en_data_path)
        self.vi_tokenizer, self.vi_vocab_size = self._create_tokenizer(vi_data_path)
        
        # Define start and end token ids
        self.start_token_id = tf.constant([self.vi_vocab_size])
        self.end_token_id = tf.constant([self.vi_vocab_size + 1])
    
    def _unicode_to_ascii(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')
    
    def _preprocess_text(self, w):
        w = self._unicode_to_ascii(w.lower().strip())
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
        w = w.strip()
        w = '<start> ' + w + ' <end>'
        return w
    
    def _create_tokenizer(self, data_path):
        # Load the data and preprocess it
        with open(data_path, 'r', encoding='utf-8') as f:
            data = f.read().splitlines()
        data = [self._preprocess_text(sentence) for sentence in data]
        
        # Create a tokenizer
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        tokenizer.fit_on_texts(data)
        
        # Return the tokenizer and the vocabulary size
        return tokenizer, len(tokenizer.word_index)
    
   

