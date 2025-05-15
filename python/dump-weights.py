#!/usr/bin/env python

import torch
from torch import nn
import numpy as np
from torchinfo import summary

import torch.nn.utils.parametrize as parametrize

from modeling_MERT import *

config = MERTConfig()
model = MERTModel(config)

dict=torch.load('../data/pytorch_model.bin')
model.load_state_dict(dict)
summary(model)

parametrize.remove_parametrizations(model.encoder.pos_conv_embed.conv, 'weight')
summary(model)

all_weights = list(model.state_dict().items())

w = { k : v for k, v in all_weights }
torch.save(w, '../data/for_libtorch.pth')

i = 0
n = len(all_weights)

for k, v in all_weights:
    print (f'{i+1}/{n}')
    with open(f'../data/weights_{i:>03}.cpp', 'w') as fout:
        fout.write(f'sizes["{k}"] = {{ ')
        first = True
        for sz in v.shape:
            if not first:
                fout.write(', ')
            fout.write(f'{sz}')
            first = False
        fout.write(' };\n')
        fout.write(f'data["{k}"] = {{ ')
        arr = v.contiguous().numpy()
        first = True
        for x in arr.flat:
            if not first:
                fout.write(',')
            fout.write(str(x))
            first = False
        fout.write(' };\n')
    i = i + 1

with open('../data/weights.hpp', 'w') as fout:
    fout.write('#include <map>\n')
    fout.write('#include <vector>\n')
    fout.write('#include <string>\n')
    fout.write('extern std::map<std::string, std::vector<int>> sizes;\n')
    fout.write('extern std::map<std::string, std::vector<float>> data;\n')
    fout.write('extern void init_weights();\n')

with open('../data/weights.cpp', 'w') as fout:
    fout.write('#include "weights.hpp"\n')
    fout.write('std::map<std::string, std::vector<int>> sizes = {};\n')
    fout.write('std::map<std::string, std::vector<float>> data = {};\n')
    fout.write('void init_weights() {\n')
    for i in range(1, n):
        fout.write(f'#include "weights_{i:>03}.cpp"\n')
    fout.write('}\n')

