# %pip install icecream
# %pip install flash_attn
# %pip install --upgarde transformers
# %pip install sentencepiece
# %pip install protobuf==4.23.3
# %pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
# %pip install einops
# %pip install flash_attn
# %pip install 'accelerate>=0.26.0'

import torch
import os
from transformers import AutoTokenizer, AutoModel
from icecream import ic
import time
class DocOwlInfer():
    def init(self, ckpt_path):
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=False)
        self.model = AutoModel.from_pretrained(ckpt_path, trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map='auto')
        self.model.init_processor(tokenizer=self.tokenizer, basic_image_size=504, crop_anchors='grid_12')

    def inference(self, images, query):
        messages = [{'role': 'USER', 'content': '<|image|>'*len(images)+query}]
        answer = self.model.chat(messages=messages, images=images, tokenizer=self.tokenizer)
        return answer


docowl = DocOwlInfer(ckpt_path='mPLUG/DocOwl2')
images = [
        r'C:\Users\Santhosh.M\Downloads\Zoho.Trident.Sideload_p2mzwbqt4767m!App\image1.jpg',
        r'C:\Users\Santhosh.M\Downloads\Zoho.Trident.Sideload_p2mzwbqt4767m!App\image2.jpg',
        r'C:\Users\Santhosh.M\Downloads\Zoho.Trident.Sideload_p2mzwbqt4767m!App\image3/jpg',
    ]

answer = docowl.inference(images, query='what is this paper about? provide detailed information.')

answer = docowl.inference(images, query='what is the third page about? provide detailed information.')
