# __SPEECH-EMOTION-RECOGNTION__

- This repo illustrates how to use Speech-emotion-recognition module with ROS
- We need Ros, audio_common and pytorch or tensorflow. we give two version with torch and tesorflow.
- requirements must be installed. 
___
## Datasets
- We use KESDy18 Korean emotion datasets.
- it includes 2880 wav files. And we only use 4 emotions. (angry, happy, sad, neutral)
- You can download datafiles in [here](https://nanum.etri.re.kr/share/kjnoh/SER-DB-ETRIv18?lang=ko_KR) after submit License Agreement.



## How to trained __(pytorch)__
- First, clone this repo.
- Train code created by jupyter notebook (python 3.8.12).
- Locate wav files in './data' and do preprocessing to csv or list.
- Select model in torchvision.models(using DenseNet in this code) and chage __input size(in_features)__ to fit the model.
    ```
    model.classifier = nn.Linear(in_features=1024, out_features=4)
## How to predict
- set wav file in './predict_audio'.  
  you must set only one file in this dir or fix the code.
   ```
    python predict_torch.py
## Reference
- benchmark link : https://github.com/MITESHPUTHRANNEU/Speech-Emotion-Analyzer
