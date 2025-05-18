import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

with open("data/the_verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")

class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_token_ids = []
        self.target_token_ids = []

        token_ids = tokenizer.encode(txt)
        
        for i in range(0, len(token_ids)-max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1: i+max_length+1]
            
            self.input_token_ids.append(torch.tensor(input_chunk))
            self.target_token_ids.append(torch.tensor(target_chunk))

    def __getitem__(self, idx):
        return self.input_token_ids[idx], self.target_token_ids[idx]
    
    def __len__(self):
        return len(self.input_token_ids)

def create_gpt_dataloader(txt, batch_size, max_length=256, stride=128, shuffle=True, drop_last=True):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer=tokenizer, max_length=max_length, stride=stride)

    return DataLoader(dataset, 
                      batch_size=batch_size, 
                      shuffle=shuffle, 
                      drop_last=drop_last
                      )


max_length = 4
gpt_dataloader = create_gpt_dataloader(raw_text, batch_size=8, max_length=max_length, stride=max_length)

iterator = iter(gpt_dataloader)
first_batch = next(iterator)

print(first_batch)