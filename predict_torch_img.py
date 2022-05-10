import os
from tkinter import image_names
import pandas as pd
import librosa
import librosa.display
import torch
import torchvision
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import numpy as np
import torch.nn as nn
import torch.optim as optim
from PIL import Image


# load model
# load_path = './saved_models/DenseNet121.pth'
load_path = './saved_models/DenseNet121_img_aihub_pretrained.pth'
load_weights = torch.load(load_path, map_location={'cuda:0': 'cpu'})
# model = torchvision.models.densenet121(pretrained=False)
model = torchvision.models.densenet121(pretrained=True)
# first_conv_layer = [nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
# first_conv_layer.extend(list(model.features))  
# model.features= nn.Sequential(*first_conv_layer )  
# model.classifier = nn.Linear(in_features=1024, out_features=4)
model.classifier = nn.Linear(in_features=1024, out_features=4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters() ,lr=0.00001, weight_decay=1e-6, momentum=0.9)
model.eval()
model.load_state_dict(load_weights)

# Preprocessing wavfile
class Testwav():
    def __init__(self, file, frame_length=0.025, frame_stride=0.010):
        self.file = file
        self.frame_length = frame_length
        self.frame_stride = frame_stride
    
    def __len__(self):
        return len(self.file)
    
    def __getitem__(self):
        audio_path = self.file
        X, sample_rate = librosa.load(audio_path, res_type='kaiser_fast',sr=16000,offset=0.0)
        sample_rate = sample_rate
        input_nfft = int(round(sample_rate*self.frame_length))
        input_stride = int(round(sample_rate*self.frame_stride))

        S = librosa.feature.melspectrogram(y=X, n_mels=64, n_fft=input_nfft, hop_length=input_stride)
        P = librosa.power_to_db(S, ref=np.max)


        return P

def getimg(data):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        p = librosa.display.specshow(data ,ax=ax, sr=16000, hop_length=int(round(16000*0.025)), x_axis='time',y_axis='linear')
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

        fig.savefig('predict.jpg' , bbox_inches=extent)
        img_path = ('predict.jpg')
        plt.ioff()
        plt.close()
        return img_path
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
class img2tensor():
    def __init__(self,data_path,transforms=test_transforms):
        self.data_path = getimg(Testwav(data_path).__getitem__())
        self.transforms = transforms

    def __len__(self):
        return len(self.data_path)

        
    def __getitem__(self):
        img_path = self.data_path
        image = Image.open(img_path)
        I = test_transforms(image)

        return I
class Predictor(object):
    def __init__(self, model, device ='cpu',  fp16=False ):
        self.model = model
        
        self.cls_name = {0:'angry', 1:'neutral', 2:'sad', 3:'happy'}
        self.device = device
        

    def predict(self, audio):
        audio_info = img2tensor(audio).__getitem__().unsqueeze(0)
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
file = './predict_audio/' + os.listdir('./predict_audio')[0]
predictor = Predictor(model)
predictor.predict(file)
