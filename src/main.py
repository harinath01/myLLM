import tokenizer

with open("/Users/testpress/myLLM/the_verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokens = tokenizer.tokenize(raw_text)
vocabulary = tokenizer.generate_vocabulary(tokens)
simple_tokenizer = tokenizer.SimpleTokenizerV1(vocabulary)

print(simple_tokenizer.encode("Hello world.<|endoftext|> in the course of world"))