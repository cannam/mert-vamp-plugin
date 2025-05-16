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

def sname(k):
    return 'sizes__' + k.replace('.', '__')

def dname(k):
    return 'data__' + k.replace('.', '__')

with open('../data/weights.hpp', 'w') as fout:
    fout.write('#pragma once\n')
    fout.write('#include <cstdint>\n')
    fout.write('#include <vector>\n')
    fout.write('#include <string>\n')
    fout.write('extern const float *lookup_model_data(std::string key, std::vector<int64_t> &sizes);\n')

with open('../data/weights.cpp', 'w') as fout:
    fout.write('#include "weights.hpp"\n')
    for k, v in all_weights:
        fout.write(f'extern const int64_t {sname(k)}[{len(v.shape)}];\n')
        fout.write(f'extern const float {dname(k)}[{v.numel()}];\n')
    fout.write('const float *lookup_model_data(std::string key, std::vector<int64_t> &sizes) {\n')
    for k, v in all_weights:
        fout.write(f'  if (key == "{k}") {{\n')
        s = sname(k)
        fout.write(f'    sizes = std::vector<int64_t>({s}, {s} + (sizeof({s})/sizeof({s}[0])));\n')
        fout.write(f'    return {dname(k)};\n')
        fout.write('  }\n')
    fout.write('  return nullptr;\n}\n')

for k, v in all_weights:
    print (f'{i+1}/{n}')
    with open(f'../data/weights_{i:>03}.cpp', 'w') as fout:
        fout.write('#include <cstdint>\n')
        fout.write(f'extern const int64_t {sname(k)}[{len(v.shape)}] = {{ ')
        first = True
        for sz in v.shape:
            if not first:
                fout.write(', ')
            fout.write(f'{sz}')
            first = False
        fout.write(' };\n')
        fout.write(f'extern const float {dname(k)}[{v.numel()}] = {{\n')
        arr = v.contiguous().numpy()
        first = True
        for x in arr.flat:
            if not first:
                fout.write(',')
            fout.write(str(x))
            first = False
        fout.write('\n};\n')
    i = i + 1

