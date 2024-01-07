from torch import nn, Tensor, optim
import torch
from typing import Optional, List
from torch.utils.data import Dataset, DataLoader

from dataset import Dataset, ByteTokenizer

from chat_models import Mirostat

def show_test(dataloader, model):
    model.eval()
    prompt = '''Instruction:
1+1等于几？

Input:


Output:
'''

    tokenizer = ByteTokenizer.from_pretrained()
    tokens = tokenizer.encode(prompt)
    y, s = model.run_tokens(tokens)
    m = Mirostat(3)
    tks = []
    t = 50
    for i in range(t):
        tk = m.choise(y.squeeze())
        tks.append(tk)
        if i <= t - 1:
            y, s = model.run_tokens([tk], s)
    print(tokenizer.decode(tks))

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    s = None
    for batch, (X, y) in enumerate(dataloader):
        pred, s = model(X, s)
        loss = loss_fn(pred, y)

        loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

        if batch % 1 == 0 or (batch) * dataloader.batch_size + len(X) == size:
            loss, current = loss.item(), (batch) * dataloader.batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        if batch % 10 == 0 or (batch) * dataloader.batch_size + len(X) == size:
            show_test(dataloader, model)
            model.train()

def run_train():
    dataset_object = Dataset("E:\\belle\\school_math_0.25M.json")

    batch_size = 1
    train_data_loader = DataLoader(dataset_object, batch_size=batch_size)

    model = NeuralNetwork(dataset_object.vocabSize)

    show_test(train_data_loader, model)

    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    epochs = 1
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_data_loader, model, loss_fn, optimizer)
    print("Done!")
    return model

class NeuralNetwork(nn.Module):

    def __init__(self, vocabSize: int, stateSize: int=1024):
        super().__init__()
        self.x = torch.zeros(1, vocabSize)
        self.vocabSize = vocabSize
        self.stateSize = stateSize
        c = vocabSize + stateSize
        self.fullyConnectedNeuralNetwork = nn.Sequential(
            nn.Linear(c, c),
            nn.Tanh(),
            nn.Linear(c, c),
            nn.Tanh(),
            nn.Linear(c, c)
        )

    def forward(self, x: Tensor, state: Optional[Tensor]=None) -> tuple:
        if 2 == len(x.size()) and 1 == x.size()[0] and self.vocabSize == x.size()[1]:
            return self._s_forward(x, state)
        else:
            y = None
            s = None
            b1, b2, b3 = x.size()
            for i1 in range(b1):
                _y = None
                for i2 in range(b2):
                    _x, s = self._s_forward(x[i1][i2].view(1, self.vocabSize), s)
                    if _y is None:
                        _y = _x
                    else:
                        _y = torch.cat([_y, _x], 0)
                if y is None:
                    y = _y.view(1, b2, self.vocabSize)
                else:
                    y = torch.cat([y, _y.view(1, b2, self.vocabSize)])
            return y, s

    def run_tokens(self, tokens: List[str], state: Optional[Tensor]=None) -> tuple:
        y = None
        s = state
        l = len(tokens)
        for i in range(l):
            torch.zeros(1, self.vocabSize, out=self.x)
            self.x[0][tokens[i]] = 1.
            y, s = self._s_forward(self.x, s)
        return y, s

    def _s_forward(self, x: Tensor, state: Optional[Tensor]=None) -> tuple:
        inputs = torch.cat([x, torch.zeros(1, self.stateSize) if state is None else state], 1)
        networkOutputs = self.fullyConnectedNeuralNetwork(inputs)
        logits, newState = torch.tensor_split(networkOutputs, [self.vocabSize, ], 1)
        return logits, newState

if __name__ == '__main__':
    torch.manual_seed(42)
    torch.set_printoptions(precision=2, sci_mode=False)

    run_train()
