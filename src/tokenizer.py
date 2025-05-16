import re

def tokenize(raw_text):
    preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
    tokens = [item.strip() for item in preprocessed if item.strip()]

    return tokens



