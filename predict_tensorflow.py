
import os
import pandas as pd
import librosa
import glob 
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import LabelEncoder
import time

lb = LabelEncoder()
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("saved_models/Emotion_DenseNet121.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
opt = RMSprop(lr=0.00001, decay=1e-6)
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
y = ['angry','happy','neutral','sad']
lb = LabelEncoder()
lb.fit(y)


def predicts(wav_path):
    start_time = time.time()
    X, sample_rate = librosa.load(wav_path, res_type='kaiser_fast',duration=2.5,sr=16000,offset=0.0)
    signal = np.zeros((48000))
    signal[:len(X)] = X
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=32)
    live=tf.expand_dims(mfccs,axis=0)
    
    livepreds = loaded_model.predict(live,batch_size=32,verbose=1)
    
    livepreds1=livepreds.argmax(axis=1)
    liveabc = livepreds1.astype(int).flatten()
    livepredictions = (lb.inverse_transform((liveabc)))
    FPS = 1.0 / (time.time() - start_time)
    print(FPS)
    print(livepredictions)
    return livepredictions



predicts('./5e378372dbc4b7182a6a9922.wav')