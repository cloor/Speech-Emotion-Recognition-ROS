# __SPEECH-EMOTION-RECOGNTION__

- This repo illustrates how to use Speech-emotion-recognition module with ROS
- We need Ros, audio_common and pytorch.
- requirements must be installed. 
___
# Datasets
- ## __KESDy18__
  - We use KESDy18 Korean emotion datasets.
  - This includes 2880 wav files. And we only use 4 emotions. (0 = angry, 1 = neutral, 2 = sad, 3 = happy)
  - You can download datafiles in [here](https://nanum.etri.re.kr/share/kjnoh/SER-DB-ETRIv18?lang=ko_KR) after submit License Agreement.
- ## __AIHUB__
  - We use Aihub Korean emotion datasets.
  - This includes about 50,000 wav files with text, ages, ...
  - You cna download datafiles in [here](https://aihub.or.kr/opendata/keti-data/recognition-laguage/KETI-02-002)
___
# Feature Extraction & Model
- ## Feature Extraction
  - For feature extration we make use of the LIBROSA library
___
# How to use 
- ## How to trained __(pytorch)__
  - First, clone this repo.
  - Train code created by jupyter notebook (python 3.8.12).
  - Locate wav files in './data' and do preprocessing to csv or list.
  - Select model in torchvision.models(using DenseNet in this code) and chage __input size(in_features)__ to fit the model.
      ```
      model.classifier = nn.Linear(in_features=1024, out_features=4)
      ```
- ## How to record
  - run record_4sec.py -> .wav file will saved in './predict_audio'
  - ```
    python record_4sec.py
    ```
- ## How to predict
  - set wav file in './predict_audio'.  
    you must set only one file in this dir or fix the code.
  - ```
    python predict_torch_img.py
## Reference
- benchmark link : https://github.com/MITESHPUTHRANNEU/Speech-Emotion-Analyzer