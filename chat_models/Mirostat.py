# MIT License

# Copyright (c) 2021 Sourya Basu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import torch
from torch import Tensor

def estimate_s(prob, top_s: int=100):
    result = 0
    num = 0
    den = 0
    for i in range(top_s):
        b = prob[i]/prob[i+1]
        t = (i+2)/(i+1)
        num += math.log(b)*math.log(t)
        den += math.log(t)**2
    return num/den


def compute_k(n,s,tau):
    eps = s-1
    k = ((eps*(2**(tau)))/(1-n**(-eps)))**(1/s)
    k = round(k)
    return k

class Mirostat(object):

    def __init__(self, tau: float=3.0, lr: float=1.0):
        self.mirostat_tau = tau
        self.max_surprise = 2 * tau
        self.learning_rate = lr

    def choise(self, out: Tensor) -> int:
        if len(out.shape) != 1:
            raise TypeError(
                'Argument `out` passed to choise() must be 1d Tensor, %sd given'%(len(out.shape))
            )
        n = len(out)
        sorted_logits, sorted_indices = torch.sort(out, descending=True)
        prob_original = torch.softmax(sorted_logits, dim=-1).tolist()

        s = estimate_s(prob_original)
        k = compute_k(n,s,self.max_surprise) + 1

        sorted_logits = sorted_logits[0:k]
        sorted_indices = sorted_indices[0:k]

        prob_topk = torch.softmax(sorted_logits, dim = 0)

        prev_i = torch.multinomial(prob_topk, num_samples=1, replacement=True)

        index_surprise = math.log2(1/prob_original[prev_i])

        prev = sorted_indices[prev_i]

        error_surprise = index_surprise - self.mirostat_tau
        self.max_surprise -= self.learning_rate*error_surprise
        return int(prev[0])
