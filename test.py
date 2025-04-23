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

model.eval()

print(model)

scripted_model = torch.jit.script(model)
scripted_model.save("scripted_model.pt")

audio, file_rate = librosa.load('stairway-intro.wav', sr = 16000, mono = True)
t_audio = torch.from_numpy(np.array([audio]))

with torch.no_grad():
    outputs = model(t_audio, output_hidden_states=True)
    
