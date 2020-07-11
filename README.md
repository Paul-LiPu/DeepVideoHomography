# DeepHomography
This repository is the implementation of our ICPR 2020 submission:
Learning Knowledge-Rich Sequence-to-SequenceModel for Planar Homograph Estimation in AerialVideos

### <a name="dependency"></a> Dependency
* Ubuntu ≥ 14.04
* Python ≥ 3.6.8
* CUDA ≥ 8.0
* Pytorch == 1.0.0
opencv-python==3.4.2.17　opencv-contrib-python==3.4.2.17


Other python libraries:
> ```bash
> pip install -r requirements.txt
> ```


### <a name="data"></a> Data
The raw data can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1e6oG_4b4HbBA-rNfMknxShBR1ITVrckh?usp=sharing). 

### <a name="data_preprocessing"></a> Data Preprocessing
We did three things:
1. Transfer audio files to spectrogram images. 
2. Transfer ground truth files to binary black/white images with the same size of corresponding spectrogram.
3. Crop images into patches and stored in HDF5 files.

To transfer raw data into images, you may run the code with:
> ```bash
> python 1.Spectrogram_and_GT/generate_traindata.py --audio_dir PATH_TO_AUDIO_FILES  \ 
>   --annotation_dir PATH_TO_ANNOTATION_FILES --output_dir PATH_TO_OUTPUT_SPECTROGRAM
> ```

To generate training HDF5 files, you may configure the paths within 
matlab code file 2.Network_Training_data/generate_train_hdf5.m, and run the matlab file.

### <a name="model_training"></a> Model training
python video2image_resize.py --input_path /data2/homography_estimation/data/test/videos --output_path /data2/homography_estimation/data/test/images
python generate_testdata.py --data_path /data2/homography_estimation/data/test/images --patch_select center --scale 0.25 --height 720 --width 1280 --patch_size 128 --output_dir /data2/homography_estimation/data/test/data2  
python generate_testdata.py --data_path /data2/homography_estimation/data/test/images --patch_select center --scale 0.25 --height 720 --width 1280 --patch_size 128 --output_dir /data2/homography_estimation/data/test/data --regenerate False
python generate_traindata.py --data_path /data2/homography_estimation/data/test/images --patch_select random --scale 0.25 --height 720 --width 1280 --patch_size 128 --split 0.9 --n_frame 2 --n_patch 1 --pattern "clip(\d+)" --output_dir /data2/homography_estimation/data/test/train_data
python generate_traindata.py --data_path /data2/homography_estimation/data/Freeway/train_images --patch_select random --scale 0.25 --height 720 --width 1280 --patch_size 128 --split 0.9 --n_frame 2 --n_patch 1 --pattern "clip_(\d+)_.+" --output_dir /data2/homography_estimation/data/test/train_data
python test_homography_opencv.py --data_path /data2/homography_estimation/data/test/data --method Identity --ann_path /data2/homography_estimation/data/test/ann --scale 0.25
python test_homography_network.py --model_type CNN --model_file models/BASE/HCNN-final.pth --data_path /data2/homography_estimation/data/test/data --ann_path /data2/homography_estimation/data/test/ann --scale 0.25
python test_homography_network.py --model_type LSTM --model_file models/LSTM/HCNN-final.pth --data_path /data2/homography_estimation/data/test/data --ann_path /data2/homography_estimation/data/test/ann --scale 0.25



1. Train DWC-I model
> ```bash
> python 3.Network_train_and_test/train.py --data_type h5 \ 
>   --train_data 3.Network_train_and_test/train_DWC-I.txt \
>   --test_data 3.Network_train_and_test/test.txt \
>   --exp_name DWC-I
> ```
2. Train DWC-II model
> ```bash
> python 3.Network_train_and_test/train.py --data_type lmdb \ 
>   --train_data 3.Network_train_and_test/train_DWC-II.txt \
>   --test_data 3.Network_train_and_test/test.txt \
>   --exp_name DWC-II
> ```
3. Train DWC-III model
> ```bash
> python 3.Network_train_and_test/train.py --data_type lmdb \ 
>   --train_data 3.Network_train_and_test/train_DWC-III.txt \
>   --test_data 3.Network_train_and_test/test.txt \
>   --exp_name DWC-III
> ```
4. Train DWC-IV model
> ```bash
> python 3.Network_train_and_test/train.py --data_type lmdb \ 
>   --train_data 3.Network_train_and_test/train_DWC-IV.txt \
>   --test_data 3.Network_train_and_test/test.txt \
>   --recall_guided True \
>   --recall_val_data 3.Network_train_and_test/val.txt  \
>   --pretrained_model 3.Network_train_and_test/models/DWC-III.pth \
>   --exp_name DWC-IV
> ```

### <a name="model_training"></a> Model inference
To generate confidence map for each spectrogram, you may run:
> ```bash
> python 3.Network_train_and_test/test.py \ 
>   --model_file PATH_TO_YOUR_MODEL \
>   --test_img_dir PATH_TO_YOUR_TEST_IMAGES \
>   --output_dir PATH_TO_YOUR_OUTPUT
> ```
For example, you can run the following code to get confidence maps from DWC-I on our testing dataset. 
> ```bash
> python 3.Network_train_and_test/test.py \ 
>   --model_file 3.Network_train_and_test/models/DWC-I.pth \
>   --test_img_dir data/test_imgs \
>   --output_dir 3.Network_train_and_test/test_results/DWC-I
> ```

### <a name="model_evaluation"></a> Model Evaluation
To extract whistles from confidence map, and evaluate the performance. You may use our modified silbido, and 
run the following code in MATLAB:
> ```bash
> silbido_init
> test_detection_score_batch
> ```

Please find the original silbido package [here](https://roch.sdsu.edu/index.php/software/). 
