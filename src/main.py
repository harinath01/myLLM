import tokenizer

with open("/Users/testpress/myLLM/the_verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokens = tokenizer.tokenize(raw_text)
print(tokens)