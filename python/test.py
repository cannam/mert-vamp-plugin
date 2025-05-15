#!/usr/bin/env python

import torch
from torch import nn
import numpy as np
import librosa
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

w = { k : v for k, v in model.state_dict().items() }
torch.save(w, '../data/for_libtorch.pth')

print('keys are:')
print(model.state_dict().keys())

model.eval()

print(model)

#scripted_model = torch.jit.script(model)
#scripted_model.save("scripted_model.pt")

#audio, file_rate = librosa.load('stairway-intro-16k-mono.wav', sr = 16000, mono = True)
audio, file_rate = librosa.load('../data/gerudo.wav', sr = 16000, mono = True)
t_audio = torch.from_numpy(np.array([audio]))

with torch.no_grad():
    outputs = model(t_audio, output_hidden_states=True)

all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
print(all_layer_hidden_states.shape) # [13 layer, Time steps, 768 feature_dim]

import pandas as pd

l0_np = all_layer_hidden_states[0].numpy()
df = pd.DataFrame(l0_np)
df.to_csv("out_0.csv", index=False)

l12_np = all_layer_hidden_states[12].numpy()
df = pd.DataFrame(l12_np)
df.to_csv("out_12.csv", index=False)

