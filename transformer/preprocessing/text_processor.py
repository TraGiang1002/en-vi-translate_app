import tensorflow as tf

class TextProcessor:
    def __init__(self, corpus):
        self.corpus = corpus
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        self.vocab_size = None
        self.word2idx = None
        self.idx2word = None
        self.max_len = None
        self._build_vocab()

    def _build_vocab(self):
        self.tokenizer.fit_on_texts(self.corpus)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.word2idx = self.tokenizer.word_index
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.max_len = max([len(sentence.split()) for sentence in self.corpus])

    def encode_sentence(self, sentence):
        encoded_sentence = self.tokenizer.texts_to_sequences([sentence])[0]
        return encoded_sentence

    def decode_sentence(self, sequence):
        sentence = ' '.join([self.idx2word[idx] for idx in sequence if idx != 0])
        return sentence
