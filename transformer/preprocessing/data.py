import os
from transformers import AutoTokenizer

def load_data(en_file, vi_file):
    with open(en_file, "r", encoding="utf-8") as f:
        en_data = f.readlines()
    with open(vi_file, "r", encoding="utf-8") as f:
        vi_data = f.readlines()
    return en_data, vi_data

def preprocess_data(en_data, vi_data, tokenizer):
    preprocessed_en_data = []
    preprocessed_vi_data = []
    for en_sent, vi_sent in zip(en_data, vi_data):
        en_sent = en_sent.strip().lower()
        vi_sent = vi_sent.strip().lower()

        en_tokens = tokenizer.tokenize(en_sent)
        vi_tokens = tokenizer.tokenize(vi_sent)

        en_tokens = ["[CLS]"] + en_tokens + ["[SEP]"]
        vi_tokens = ["[CLS]"] + vi_tokens + ["[SEP]"]

        en_ids = tokenizer.convert_tokens_to_ids(en_tokens)
        vi_ids = tokenizer.convert_tokens_to_ids(vi_tokens)

        preprocessed_en_data.append(en_ids)
        preprocessed_vi_data.append(vi_ids)

    return preprocessed_en_data, preprocessed_vi_data

def save_data(en_data, vi_data, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, "en_data.txt"), "w", encoding="utf-8") as f:
        for en_sent in en_data:
            f.write(" ".join([str(token) for token in en_sent]) + "\n")
    
    with open(os.path.join(output_dir, "vi_data.txt"), "w", encoding="utf-8") as f:
        for vi_sent in vi_data:
            f.write(" ".join([str(token) for token in vi_sent]) + "\n")

def main():
    en_data, vi_data = load_data("data/en_sents.en", "data/vi_sents.vi")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    preprocessed_en_data, preprocessed_vi_data = preprocess_data(en_data, vi_data, tokenizer)
    save_data(preprocessed_en_data, preprocessed_vi_data, "data/preprocessed")

if __name__ == "__main__":
    main()
