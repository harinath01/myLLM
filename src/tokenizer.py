import re

def tokenize(raw_text):
    preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
    tokens = [item.strip() for item in preprocessed if item.strip()]

    return tokens

def generate_vocabulary(tokens):
    unique_tokens = sorted(set(tokens))
    
    return {token:id for id,token in enumerate(unique_tokens) }


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {id: token for id, token in enumerate(vocab)}

    def encode(self, sentence):
        tokens = tokenize(sentence)

        return [self.str_to_int[token] for token in tokens]

    def decode(self, ids):
        texts =  [self.int_to_str[id] for id in ids]
        return " ".join(texts)
