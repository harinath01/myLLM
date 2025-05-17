import tiktoken

with open("/home/hari/workspace/myLLM/the_verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")
tokens = tokenizer.encode(raw_text)

context_size = 10
for i in range(1, context_size + 1):
    inputs = tokens[:i]
    target = tokens[i]
    print(tokenizer.decode(inputs), "---->", tokenizer.decode([target]))

