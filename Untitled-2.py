#import kagglehub
#path = kagglehub.dataset_download("golammostofas/the-verdict")
#print("Path to dataset files:", path)

with open("C:/Users/Santhosh.M/Documents/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("total_number_of_character",len(raw_text))
print(raw_text[:99])

import re
text = "hello, world. this, is a test."
result = re.split(r'(\s)',text)
print(result)
result = re.split(r'([,.]|\s)',text)
print(result)
#removing the white spaces
result = [item for item in result if item.strip()]
print(result)

text="hello, world. is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)',text)
result = [item for item in result if item.strip()]
print(result)

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item for item in preprocessed if item.strip()]
print(preprocessed[:30])
print(len(preprocessed))

all_words = sorted(set(preprocessed))
vocal_size = len(all_words)
print(vocal_size)

vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i>=50:
        break   

class SimpleTokenizerV1:
    def __init__(self,vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed=re.split(r'([,.:;?_!"()\']|--|\s)',text)
        preprocessed=[item for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self,ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        #replaced the words with specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])',r'\1',text)
        return text
    
tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know."
            Mrs.Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

#token ids back into texts
output=tokenizer.decode(ids)
print(output)

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>","<|unk|>"])
vocab = {token:integer for integer,token in enumerate (all_tokens)}

output1=len(vocab.items())
print(output1)
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)


class SimpleTokenizerV2:
    def __init__(self,vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    
    def encode(self,text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)',text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed]
        ids=[self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self,ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        #replaced the words with specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])',r'\1',text)
        return text
tokenizer = SimpleTokenizerV2(vocab)
text1 = "hello, do you like tea?"
text2 = "In the sunlight terraces of the place."
text = "<|endoftext|>".join((text1,text2))
print(text)

output2 = tokenizer.encode(text)
print(output2)

final_output = tokenizer.decode(tokenizer.encode(text))
print(final_output)