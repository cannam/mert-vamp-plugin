#!/usr/bin/env python

from transformers import Wav2Vec2FeatureExtractor
import torch
from torch import nn
import numpy as np
import librosa
from torchinfo import summary

from modeling_MERT import *

config = MERTConfig()
model = MERTModel(config)

dict=torch.load('pytorch_model.bin')
model.load_state_dict(dict)
summary(model)

torch.save(model.state_dict(), 'for_libtorch.pth')

w = { k : v for k, v in model.state_dict().items() }
torch.save(w, 'fart.pth')

print('keys are:')
print(model.state_dict().keys())

model.eval()

print(model)

#scripted_model = torch.jit.script(model)
#scripted_model.save("scripted_model.pt")

audio, file_rate = librosa.load('stairway-intro.wav', sr = 16000, mono = True)
t_audio = torch.from_numpy(np.array([audio]))

with torch.no_grad():
    outputs = model(t_audio, output_hidden_states=True)

all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()
print(all_layer_hidden_states.shape) # [13 layer, Time steps, 768 feature_dim]

import pandas as pd

l12_np = all_layer_hidden_states[12].numpy()
df = pd.DataFrame(l12_np)
df.to_csv("out.csv", index=False)

