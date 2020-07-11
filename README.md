# Deep Video Homography Estimation
This repository is the implementation of our ICPR 2020 submission:
Learning Knowledge-Rich Sequence-to-Sequence Model for Planar Homography Estimation in Aerial Videos

### <a name="dependency"></a> Dependency
* Ubuntu ≥ 14.04
* Python ≥ 3.6.8
* CUDA ≥ 8.0
* Pytorch == 1.0.0
* opencv-python==3.4.2.17
* opencv-contrib-python==3.4.2.17

Please make sure to use the correct version of Pytorch, opencv-python, opencv-contrib-python. The code might be 
incompatible with the lastest version of those libraries. 

Other python libraries:
> ```bash
> pip install -r requirements.txt
> ```

### <a name="data"></a> Data
The dataset can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1e6oG_4b4HbBA-rNfMknxShBR1ITVrckh?usp=sharing). 
Folder "train_data" and "test_data" contains 141 and 22 aerials videos for training and testing respectively. Folder "test_data"
also includes the point correspondence annotations of the testing videos. Each line in the annotation files contains two 
correspondent points in different frames. Each point is recorded in the format "[Frame number] [Horizontal coordinate] [Vertical coordinate]". 
The frame number are 0-based, and the coordinate system use the left-upper corner in image as origin. 

### <a name="data_preprocessing"></a> Data Preprocessing
Two steps are used for preprocessing the videos into training or testing data. 
1. Extract frames from videos. 
2. Resize the image and generate the 

To generate the testing data, you may run the code:
> ```bash
> python video2image_resize.py --input_path [TEST_VIDEOS_FOLDER] \
>                              --output_path [TEST_FRAME_IMAGE_FOLDER]
> python generate_testdata.py --data_path [TEST_FRAME_IMAGE_FOLDER] \
>                             --patch_select center \
>                             --scale 0.25 \
>                             --height 720 \
>                             --width 1280 \
>                             --patch_size 128 \
>                             --output_dir [TEST_DATA_FOLDER] 
> ```

To generate the training data, you may run the code:
> ```bash
> python video2image_resize.py --input_path [TRAIN_VIDEOS_FOLDER] \
>                              --output_path [TRAIN_FRAME_IMAGE_FOLDER]
> python generate_traindata.py --data_path [TRAIN_FRAME_IMAGE_FOLDER] \
>                             --patch_select random \
>                             --scale 0.25 \
>                             --height 720 \
>                             --width 1280 \
>                             --patch_size 128 \
>                             --split 0.9 \
>                             --n_frame 2 \
>                             --n_patch 1 \
>                             --output_dir [TRAIN_DATA_FOLDER] 
> ```
Arguments n_frame and n_patch may vary in different experiments. The training details will 
be updated later. 


### <a name="trained_model"></a> Trained Model
The trained model for our experiments in the paper could be downloaded in [Google Drive](https://drive.google.com/drive/folders/1e6oG_4b4HbBA-rNfMknxShBR1ITVrckh?usp=sharing),
which is in the folder "trained_model". Each subfolder which contains the model weights file has name corresponded to our experiment and Table 1 in the paper.
 

### <a name="model_evaluation"></a> Model Evaluation
1. To reproduce the evaluation traditional homography estimation methods in our paper:
> ```bash
> python test_homography_opencv.py --data_path [TEST_DATA_FOLDER] \
>                                  --method [ESTIMATION_METHOD] \
>                                  --ann_path [TEST_ANNOTATION_FOLDER] 
>                                  --scale 0.25
> ```
ESTIMATION_METHOD could be Identity or ORB2+RANSAC, which is correponded to our Table 1 in the paper.  

2 To reproduce the evaluation of deep models in our paper:
> ```bash
> python test_homography_network.py --model_type [MODEL_TYPE] 
>                                   --model_file [PATH_TO_MODEL_WEIGHTS]
>                                   --data_path [TEST_DATA_FOLDER] 
>                                   --ann_path [TEST_ANNOTATION_FOLDER]  
>                                   --scale 0.25
> ```
MODEL_TYPE should be CNN for model BASE, REG-P, REG-S, REG-T, and REG-ALL. MODEL_TYPE should be LSTM for model LSTM, LSTM-REG-ALL. 


### <a name="model_training"></a> Model training
The model training code will be released later. 
