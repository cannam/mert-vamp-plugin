#!/usr/bin/env python

import torch
from torch import nn
import numpy as np
import librosa

from modeling_MERT import *

config = MERTConfig()
model = MERTModel(config)

dict=torch.load('../ext/MERT-v1-95M/pytorch_model.bin')
model.load_state_dict(dict)

model.eval()

print(model)

audio, file_rate = librosa.load('../data/testfile.wav', sr = 16000, mono = True)
t_audio = torch.from_numpy(np.array([audio]))

with torch.no_grad():
    outputs = model(t_audio, output_hidden_states=True)

all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
print(all_layer_hidden_states.shape) # [13 layer, Time steps, 768 feature_dim]

import pandas as pd

l12_np = all_layer_hidden_states[12].numpy()
df = pd.DataFrame(l12_np)
df.to_csv("out-12-pytorch.csv", index=False)

