import os
import pandas as pd
import librosa
import torch
import torchvision
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim

# load model
load_path = './saved_models/DenseNet121.pth'
load_weights = torch.load(load_path, map_location={'cuda:0': 'cpu'})
model = torchvision.models.densenet121(pretrained=False)
first_conv_layer = [nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
first_conv_layer.extend(list(model.features))  
model.features= nn.Sequential(*first_conv_layer )  
model.classifier = nn.Linear(in_features=1024, out_features=4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters() ,lr=0.00001, weight_decay=1e-6, momentum=0.9)
model.eval()
model.load_state_dict(load_weights)

# Preprocessing wavfile
class Testwav():
    def __init__(self, file):
        self.file = file
    
    def __len__(self):
        return len(self.file)
    
    def __getitem__(self):
        ## MFCC
        audio_path = self.file
        X, sample_rate = librosa.load(audio_path, res_type='kaiser_fast',duration=2.5,sr=16000,offset=0.0)
        signal = np.zeros((int(sample_rate *3,)))
        signal[:len(X)] = X
        sample_rate = sample_rate
        mfccs = librosa.feature.mfcc(y=signal, 
                                            sr=sample_rate, 
                                            n_mfcc=32,
                                            )
        mfccs = torch.Tensor(mfccs)
        mfccs = mfccs.unsqueeze(0)
        mfccs = mfccs.unsqueeze(0)

        return mfccs

# Predict
class Predictor(object):
    def __init__(self, model, device ='cpu',  fp16=False ):
        self.model = model
        
        self.cls_name = {0:'angry', 1:'happy', 2:'sad', 3:'neutral'}
        self.device = device
        

    def predict(self, audio):
        
        audio_info = Testwav(audio).__getitem__()
        outputs = self.model(audio_info)
        probability = torch.softmax(outputs,1)
        probability = probability.squeeze()
        proba, idx = torch.max(probability, dim=0)
        emo_proba = proba.item()
        print(emo_proba)
        idx = idx.item()
        emo_label = self.cls_name[idx]
        print(emo_label)
        return emo_label


predictor = Predictor(model)
file = './predict_audio/' + os.listdir('./predict_audio')[0]
predictor.predict(file)