#!/usr/bin/env python

from dataclasses import replace
from glob import glob
import os
from tkinter import image_names
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
from std_msgs.msg import String
import sys
import rospy
from audio_common_msgs.msg import AudioData
from audio_common_msgs.msg import Audio_Result
import wave
import numpy as np
import soundfile as sf
from queue import Queue
import threading

# load model
load_path = '/home/seojungin/catkin_ws/src/speech_emotion/scripts/saved_models/DenseNet121_img_aihub+custom+augmentaion.pth'
load_weights = torch.load(load_path, map_location={'cuda:0': 'cpu'})
model = torchvision.models.densenet121(pretrained=True)
model.classifier = nn.Linear(in_features=1024, out_features=4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters() ,lr=0.00001, weight_decay=1e-6, momentum=0.9)
model.eval()
model.load_state_dict(load_weights)

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# publish node
result_pub = rospy.Publisher('audio_recognition_result', Audio_Result)
frames=[]

class Testwav():
    def __init__(self, file, frame_length=0.025, frame_stride=0.010):
        # self.file = file
        self.X = file
        self.frame_length = frame_length
        self.frame_stride = frame_stride
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self):
        # audio_path = self.file
        # X, sample_rate = librosa.load(audio_path, res_type='kaiser_fast',sr=16000,offset=0.0)
        sample_rate = 16000
        input_nfft = int(round(sample_rate*self.frame_length))
        input_stride = int(round(sample_rate*self.frame_stride))

        S = librosa.feature.melspectrogram(y=self.X, n_mels=64, n_fft=input_nfft, hop_length=input_stride)
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
        return emo_label, emo_proba

class UserDataManager(object):
    def __init__(self):
        self.__record_start = False
        self.__speaking_buffer = np.array([])

    def record_start(self):
        self.__record_start = True
        self.__speaking_buffer = np.array([])
        
    def add_sound(self, msg):
        global flag, flag_2
        rospy.Subscriber('/audio_recognition_msg',String, message_callback)
        predictor = Predictor(model)
        
        if self.__record_start is True:
            if flag_2 == 0:
                data = msg.data
                data = np.frombuffer(data, dtype=np.int16)
                data = np.nan_to_num(data)
                data_float = self.__pcm2float(data)
                self.__speaking_buffer = np.concatenate((self.__speaking_buffer, data_float))
                print(self.__speaking_buffer.shape)
            else:
                print('save')
                # self.result = speaking_buffer
                
                recog_result = predictor.predict(self.__speaking_buffer)
                result = Audio_Result()
                result.emotion = str(recog_result[0])
                result.confidence_rate = str(recog_result[1])
                result_pub.publish(result)
                self.save('test.wav')

                self.__speaking_buffer = np.array([])
                # rospy.on_shutdown(self.myhook)
                self.__record_start = False
    def myhook():
        print ("shutdown time!")
    
    def save(self, f_name):
        print("Save!!")
        sf.write(f_name, self.__speaking_buffer, 16000, format='wav')
        # self.__record_start = False

    def __pcm2float(self, sig, dtype='float32'):
        sig = np.asarray(sig)
        if sig.dtype.kind not in 'iu':
            raise TypeError("'sig' must be an array of integers")
        dtype = np.dtype(dtype)
        if dtype.kind != 'f':
            raise TypeError("'dtype' must be a floating point type")

        i = np.iinfo(sig.dtype)
        abs_max = 2 ** (i.bits - 1)
        offset = i.min + abs_max
        return (sig.astype(dtype) - offset) / abs_max

    # def data(self):

    #   result = self.result
    #         return result


# audio captrue to wav
def soundcapture():
    global  flag, flag_2
    user_data = UserDataManager()
    user_data.record_start()
    
    rospy.Subscriber('/audio_recognition_msg',String, message_callback)
    rospy.Subscriber("/audio", AudioData, user_data.add_sound)
    rospy.spin()
    
        
        


def audiocaptrue():
    # audio = rospy.wait_for_message('/audio',AudioData)
    
    rospy.Subscriber('/audio_recognition_msg', String, message_callback)
    predictor = Predictor
    rospy.Subscriber("/audio", AudioData, callback,frames)
    user_data = UserDataManager()
    
    
        
    rospy.sleep(2)
    
    file = wave.open("audio.wav", "wb")

    file.setnchannels(1)
    file.setsampwidth(2)
    file.setframerate(16000)
    file.writeframes(b''.join(frames))
    file.close()
    if frames != []:
        predictor = Predictor(model)
        recog_result = predictor.predict('audio.wav')
    try:
        result = Audio_Result()
        result.emotion = str(recog_result[0])
        result.confidence_rate = str(recog_result[1])
        result_pub.publish(result)
    except:
        pass

#def callback(data):
#    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
def callback(msg, queue):
    data = np.frombuffer(msg.data, dtype=np.int16)
    data = np.nan_to_num(data)
    sig = np.asarray(data)
    if sig.dtype.kind not in 'iu':
            raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype('float32')
    if dtype.kind != 'f':
            raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    data_float = (sig.astype(dtype) - offset) / abs_max    
    queue.append(data_float)
    

def message_callback(message):
    
    global flag,flag_2

    if message == String("on"):
        flag = 1
        flag_2 = 0
        print('flag = %d' % flag)
        print('flag_2 = %d' % flag_2)
    elif message ==String('end'):
        
        flag_2 = 1
        
    elif message == String('reset'):
        print('flag = %d' % flag)
        flag = 0
        flag_2 = 0



if __name__ == '__main__':
    global flag, flag_2
    flag = 0
    flag_2 = 0
    rospy.init_node('audio_node',anonymous=True)
    
    while True: 
        rospy.Subscriber('/audio_recognition_msg',String, message_callback)
        if flag == 1:
            try:           
                    # audiocaptrue()
                soundcapture()
                        
            except rospy.ROSInterruptException:
                pass
