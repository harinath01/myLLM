import tokenizer

with open("/Users/testpress/myLLM/the_verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokens = tokenizer.tokenize(raw_text)
vocabulary = tokenizer.generate_vocabulary(tokens)
simple_tokenizer = tokenizer.SimpleTokenizerV1(vocabulary)

while True:
    token_ids = input("Enter tokens to decode\n")
    preprocessed_token_ids = [int(token_id) for token_id in token_ids.split(",")if token_id.strip()]
    print("Tokenized result: '%s'" %simple_tokenizer.decode(preprocessed_token_ids))