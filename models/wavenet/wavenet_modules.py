import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable, Function
import numpy as np


class DilatedQueue:
    def __init__(self, max_length, data=None, dilation=1, num_deq=1, num_channels=1, device=torch.device("cpu")):
        self.in_pos = 0
        self.out_pos = 0
        self.num_deq = num_deq
        self.num_channels = num_channels
        self.dilation = dilation
        self.max_length = max_length
        self.data = data
        self.device = device
        if data == None:
            self.data = torch.zeros(num_channels, max_length, device=device)

    def enqueue(self, input):
        self.data[:, self.in_pos] = input
        self.in_pos = (self.in_pos + 1) % self.max_length

    def dequeue(self, num_deq=1, dilation=1):
        start = self.out_pos - ((num_deq - 1) * dilation)
        if start < 0:
            t1 = self.data[:, start::dilation]
            t2 = self.data[:, self.out_pos % dilation:self.out_pos + 1:dilation]
            t = torch.cat((t1, t2), 1)
        else:
            t = self.data[:, start:self.out_pos + 1:dilation]

        self.out_pos = (self.out_pos + 1) % self.max_length
        return t

    def reset(self):
        self.data = torch.zeros(self.num_channels, self.max_length, device=self.device)
        self.in_pos = 0
        self.out_pos = 0


class ConstantPad1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target_size, dimension=0, value=0, pad_start=False):
        ctx.target_size = target_size
        ctx.dimension = dimension
        ctx.value = value
        ctx.pad_start = pad_start

        ctx.num_pad = target_size - input.size(dimension)
        assert ctx.num_pad >= 0, 'target size has to be greater than input size'

        ctx.input_size = input.size()

        size = list(input.size())
        size[dimension] = target_size
        output = torch.full(tuple(size), value, device=input.device)
        c_output = output

        # crop output
        if pad_start:
            c_output = c_output.narrow(dimension, ctx.num_pad, c_output.size(dimension) - ctx.num_pad)
        else:
            c_output = c_output.narrow(dimension, 0, c_output.size(dimension) - ctx.num_pad)

        c_output.copy_(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.zeros(*ctx.input_size, device=grad_output.device)
        cg_output = grad_output

        # crop grad_output
        if ctx.pad_start:
            cg_output = cg_output.narrow(ctx.dimension, ctx.num_pad, cg_output.size(ctx.dimension) - ctx.num_pad)
        else:
            cg_output = cg_output.narrow(ctx.dimension, 0, cg_output.size(ctx.dimension) - ctx.num_pad)

        grad_input.copy_(cg_output)
        return grad_input, None, None, None, None


def constant_pad_1d(input,
                    target_size,
                    dimension=0,
                    value=0,
                    pad_start=False):
    return ConstantPad1d.apply(input, target_size, dimension, value, pad_start)
