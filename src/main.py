import tokenizer
import tiktoken

with open("/Users/testpress/myLLM/the_verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokens = tokenizer.tokenize(raw_text)
vocabulary = tokenizer.generate_vocabulary(tokens)
simple_tokenizer = tokenizer.SimpleTokenizerV1(vocabulary)

# print(simple_tokenizer.encode("Hello world.<|endoftext|> in the course of world"))

gpt2_tokenizer = tiktoken.get_encoding("gpt2")

token_ids= gpt2_tokenizer.encode("Hello world, in the course of world")
print(token_ids, gpt2_tokenizer.decode(token_ids))
token_ids = gpt2_tokenizer.encode("Hari hara nathan")
for token_id in token_ids:
    print(token_id, gpt2_tokenizer.decode([token_id]))
