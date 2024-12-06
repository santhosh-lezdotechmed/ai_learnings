#byte pair encoder

import importlib
import importlib.metadata
import tiktoken

print("tiktoken version:", importlib.metadata.version("tiktoken"))
print(tiktoken.__version__)
tokenizer = tiktoken.get_encoding("gpt2")

text = ( "hello, do you like tea? <|endoftext|> In the sunrise of terraces""of someone place.")

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
import tiktoken

# List available encodings (this helps confirm the name of the tokenizer)
print(tiktoken.list_encoding_names())

strings = tokenizer.decode(integers)
print(strings)
 
integers = tokenizer.encode("Akwirw ier")
print(integers)

strings = tokenizer.decode(integers)
print(strings)

with open ("C:/Users/Santhosh.M/Documents/the-verdict.txt","r",encoding="utf-8") as f:
    raw_text = f.read()
enc_text= tokenizer.encode(raw_text)
print(len(enc_text))




from torch.utils.data import Dataset, DataLoader
class GRTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids =[]
        # tokenize the entire text
        token_ids = tokenizer.encode(txt,allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids)-max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length +1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self,idx):
        return self.input_ids[ids],self.target_ids[idx]
    
def create_dataloader_v1(txt, batch_size =4, max_length=256,stride=128, shuffle=128,shuffle=True,drop_last=True,num_workers=0):
    #intialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    #create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    #create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=shuffle,drop_last=drop_last,num_workers=num_workers)
    return dataloader
with open("","r", encoding="utf-8") as f:
    raw_text = f.read()

import torch
print("Pytorch version:", torch.__version__)

import gensim.downloader as api
model = api.load("word2vec-google-news-300")