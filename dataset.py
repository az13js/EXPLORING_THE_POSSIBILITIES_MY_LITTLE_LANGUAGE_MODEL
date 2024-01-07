from datasets import load_dataset
from torch.utils.data import Dataset as BaseDataset
import torch
from chat_models import Mirostat

from typing import Optional, List
import re

class ByteTokenizer(object):
    vocab_size: int = 256

    def from_pretrained(**args):
        return ByteTokenizer()

    def encode(self, inputString: str) -> List[int]:
        return [code for code in inputString.encode('utf-8')]

    def decode(self, tokens: List[int]) -> str:
        return bytes(tokens).decode('utf-8', errors='ignore')

class Dataset(BaseDataset):

    def __init__(self, file: str):
        self.ctx = 64
        self.squad_it_dataset = load_dataset("json", data_files=file)
        self.tokenizer = ByteTokenizer.from_pretrained()
        self.vocabSize = self.tokenizer.vocab_size
        self._role = re.compile(r'[\r|\n]+')

    def __len__(self) -> int:
        return len(self.squad_it_dataset['train'])

    def __getitem__(self, idx) -> tuple:
        item = self.squad_it_dataset['train'][idx]
        tokens = self.tokenizer.encode('Instruction:\n' + re.sub(self._role, '\n', item['instruction']).strip() + '\n\n')
        tokens = tokens + self.tokenizer.encode('Input:\n' + re.sub(self._role, '\n', item['input']).strip() + '\n\n')
        tokens = tokens + self.tokenizer.encode('Output:\n' + re.sub(self._role, '\n', item['output']).strip() + '\n\n')
        x = torch.zeros(len(tokens) - 1, self.tokenizer.vocab_size)
        y = torch.zeros(len(tokens) - 1, self.tokenizer.vocab_size)
        for i in range(len(tokens) - 1):
            x[i][tokens[i]] = 1.
            y[i][tokens[i + 1]] = 1.
        return x, y

if __name__ == '__main__':
    data = Dataset("E:\\belle\\school_math_0.25M.json")
    m = Mirostat(0.1)
    tokenizer = ByteTokenizer.from_pretrained()
    for idx, (item) in enumerate(data):
        if idx > 3:
            break
        x, y = item
        print(x)
        r = x.size()[0]
        tks = []
        for i in range(r):
            out = x[i]
            tk = m.choise(out)
            tks.append(tk)
        print(tokenizer.decode(tks))
