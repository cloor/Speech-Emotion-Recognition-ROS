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
  - We only use about 2200 data that emotion is classified clearly.
  - You can download datafiles in [here](https://aihub.or.kr/opendata/keti-data/recognition-laguage/KETI-02-002)
___
# Feature Extraction & Model
- ## Feature Extraction
  - For feature extration we make use of the LIBROSA library
    1. using mfccs to feature extraction. cut audio file in 2.5 duration and make 32 mfccs tensor shape to train in DenseNet121.
    2. using mel-spectrum make audio file to spectrum image and save it. load images to train DenseNet(pretrained=True).

  - Model
    we use DenseNet121. we choose to use densenet since model have to be light to run on 'cpu' settings.
    ## IMAGE SAMPLE
    - It's hard to see the difference.
    ### ANGRY
    ![angry](https://user-images.githubusercontent.com/88182732/167774641-fa8c135b-6c03-4aba-a306-bd4a38487093.jpg)
    ### SAD
    ![sad](https://user-images.githubusercontent.com/88182732/167774715-b74a31cd-8955-482f-bd3e-e16cfaefbeb9.jpg)

    ### NEUTRAL
    ![neutral](https://user-images.githubusercontent.com/88182732/167774683-607babcb-0c48-47c4-9caa-36bfc9d390a6.jpg)
    ### HAPPY 
    ![happy](https://user-images.githubusercontent.com/88182732/167774748-da8822b4-c27a-4d4e-baab-7f8a0f24e7e4.jpg)
    

___ 

# Train Result
- ## Result (DenseNet121)
Data                 | ETRI    | ETRI         | AIHUB       | AIHUB       |  ETRI+AIHUB | ETRI+AIHUB 
|--------------------|---------|--------------|-------------|-------------|-------------|--------------
pretrained           | False   | False        | True        | False       | True        | False       
feature extraction   | mfccs   | mel-spectrum | mel-spectrum| mel-spectrum| mel-spectrum| mel-spectrum
accuracy/(customdata)| 70%/25% | 73%/29%      | __69%/40%__ | 60%/35%     | 68%/33%     | 63%/28%
  - using mfccs in ETRI make overfitting in train data. and not good at accuracy. so we decide to use mel-spectrum. 
  - ETRI datasets also too artificial, so not fit with custom data.
  - 1. Result matrix (accuracy = 73%)
    
    ![result_matrix_img_etri](https://user-images.githubusercontent.com/88182732/167777239-b9a8b0de-5635-4cd9-9866-b1466537848d.png)

    2. Result matrix for custom data (accuracy = 40%)
    
    ![result_matrix_img_sw_aihub_pretrained](https://user-images.githubusercontent.com/88182732/167777834-a76d55bb-1874-4e83-bddd-a9c2e888dc53.png)

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
    ```
    python record_4sec.py
    ```
- ## How to predict
  - set wav file in './predict_audio'.  
    you must set only one file in this dir or fix the code.
    ```
    python predict_torch_img.py
## Reference
- benchmark link : https://github.com/MITESHPUTHRANNEU/Speech-Emotion-Analyzer