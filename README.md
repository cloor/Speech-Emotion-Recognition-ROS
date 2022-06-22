# __SPEECH-EMOTION-RECOGNTION__
This repo illustrates how to use Speech-emotion-recognition module with ROS.

We only use voice with korean, not text.

We need Ros, [audio_common](https://github.com/ros-drivers/audio_common) and pytorch. 

In audio_common_msgs, we must add Audio_result.msg and command.msg

requirements must be installed. And Ros settings also required. I use ros-kinnetic.
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

- ## __CUSTOM DATA__
  - I recorded this myself.
  - 11 sentences, with 2 levels, and 4 emotions. So 88 datafiles.

___
# Feature Extraction & Model
- ## Feature Extraction
  - For feature extration we make use of the LIBROSA library
    1. using mfccs to feature extraction. cut audio file in 2.5 duration and make 32 mfccs tensor shape to train in DenseNet121.
    2. using mel-spectrogram make audio file to spectrum image and save it. load images to train DenseNet(pretrained=True).

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
- ## Result without augmentation(DenseNet121)
  Data| Pretrained | Feature Extraction | accuracy/(custom data)
  |--------------|-------|------|-----
  ETRI|False|mfccs|70%/25%
  ETRI|False|mel-spectrogram|73%/29%
  __*AIHUB*__|__*True*__|__*mel-spectrogram*__| __*69%/40%*__
  AIHUB|False|mel-spectrogram|60%/35%
  ETRI+AIHUB | True|mel-spectrogram|68%/33%
  ETRI+AIHUB | False |mel-spectrogram|63%/28%


  - using mfccs in ETRI make overfitting in train data. and not good at accuracy. so we decide to use mel-spectrogram. 
  - ETRI dataseDts also too artificial, so not fit with custom data.
   1. Result confusion matrix (accuracy = 73%)
    
    ![result_matrix_img_etri](https://user-images.githubusercontent.com/88182732/167777239-b9a8b0de-5635-4cd9-9866-b1466537848d.png)

    2. Result confusion matrix for custom data (accuracy = 40%)
    
    ![result_matrix_img_sw_aihub_pretrained](https://user-images.githubusercontent.com/88182732/167777834-a76d55bb-1874-4e83-bddd-a9c2e888dc53.png)
    
    
- ## Result with data augmentation
     Data| Pretrained | Feature Extraction | accuracy/(custom data)
    |--------------|-------|------|-----
    __*AIHUB + CUSTOM*__| __*True*__ | __*mel-spectrogram*__ | __*86.57%/83%*__

    3. Finally we use data augmentation. Result confusion matrix(AIhub + Custom Data W augmentation)(accuracy = 83%)
    
    ![result_matrix_img_aihub+custom+augmentaion](https://user-images.githubusercontent.com/88182732/170638258-96656f58-d0f6-4321-b475-4224a828d6e2.png)
  
___ 
# How to use 
- ## How to trained __(pytorch)__
  - First, clone this repo.
  - Train code created by jupyter notebook (python 3.8.12).
  - Trainer code located in './trainer' 
  - __Locate wav files in './data' and do preprocessing to csv or list.__
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
  - Locate wav file in './predict_audio'.  
    you must locate only one file in this dir or fix the code.
    ```
    python predict_torch_img.py

- ## How to use in ROS

  1. Run audio_capture.launch 
  2. ```
      rosrun speech-emotion-ros predict.py
     ```    
  3. ```
      rosrun speech-emotion-ros command.py
     ```
  4. 'start' button to start recording, 'end' button to end recording and predict.


## Reference
- benchmark link : 
  1. https://github.com/MITESHPUTHRANNEU/Speech-Emotion-Analyzer
  2. https://github.com/Data-Science-kosta/Speech-Emotion-Classification-with-PyTorch  