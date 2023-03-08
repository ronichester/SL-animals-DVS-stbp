# SL-animals-DVS training with STBP
This repository contains an STBP (Spatio-Temporal Back Propagation) implementation on the SL-Animals-DVS dataset using Pytorch.

**A BRIEF INTRODUCTION:**  
STBP is an offline training method that directly trains a Spiking Neural Network (SNN), expanding the use of the classic backpropagation algorithm to the time domain, so the training occurs in space AND time. Therefore, it is a suitable method to train SNNs, which are biologically plausible networks (in short).  
The SL-animals-DVS is a dataset of sign language (SL) gestures peformed by different people representing animals, and recorded with a Dynamic Vision Sensor (DVS).  

<p align="center">
<img src="https://media.springernature.com/lw685/springer-static/image/art%3A10.1007%2Fs10044-021-01011-w/MediaObjects/10044_2021_1011_Fig4_HTML.png" width="300px></p>

<p align="center"> </p>  

The reported results in the SL-animals paper were a test accuracy of 56.2% +- 1.52% in the whole dataset and 71.45% +- 1.74% on the reduced dataset (excluding group S3). The results achieved with the implementation published here, using train/test sets only: **Test Accuracy (whole dataset): 41.32% +- 3.06%**.  
           
## Requirements:
While not sure if the list below contains the actual minimums, it will run for sure if you do have the following:
- Python 3.0+
- Pytorch 1.11+
- CUDA 11.3+
- python libraries: os, numpy, matplotlib, pandas, sklearn, datetime, tonic, tensorboardX

## README FIRST
This package contains the necessary python files to train a Spiking Neural Network with the STBP method on the Sign Language Animals DVS dataset. 

**IMPLEMENTATION**  
Package Contents:  
- dataset.py
- layers.py
- model.py
- sl_animals_stbp.py
- slice_data.py
- stbp_tools.py
- train_test_only.py

The SL-Animals-DVS dataset implementation code is in *dataset.py*, and it's basically a Pytorch Dataset object. The library [*Tonic*](https://tonic.readthedocs.io/en/latest/index.html#) was used to read and process the DVS recordings.

A script was created to slice the SL animals DVS recordings into actual samples for training, and save the slices to disk - *slice_data.py*. The reason for this is because the original raw dataset after download contains only 59 files (DVS recordings), and not 1121 samples. Each recording contains 1 individual performing the 19 gestures in sequence, so there is a need to manually cut these 19 slices from each whole recording in order to actually use the dataset. 

The core of the STBP method implementation is in *layers.py*: the base code code is from [thiswinex/STBP-simple](https://github-com.translate.goog/thiswinex/STBP-simple?_x_tr_sl=auto&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=wapp), to which I added a few fixes, changes and adaptations, inspired also by this other [STBP implementation](https://github.com/yjwu17/STBP-for-training-SpikingNN#spatio-temporal-bp-for-spiking-neural-networks). The main simulation parameters are in the variable *args* located in the header of *layers.py*.

The Spiking Neural Network model is in *model.py* (SLANIMALSNet), and reproduces the architecture described in the SL-animals paper. The main training tools and functions used in the package are in *stbp_tools.py*. 

The main program is in *sl_animals_slayer.py*, which contains right at the top main parameters that can be customized like 'batch size', 'data path', 'seed' and many others. This program uses the correct experimental procedure for training a network using cross validation after dividing the dataset into train, validation and test sets. A simpler version of the main program is in *train_test_only.py*, which is basically the same except dividing the dataset only into train and test sets, in an effort to replicate the published results. Apparently, the benchmark results were reported in this simpler dataset split configuration, which is not optimal.

## Use
1. Clone this repository:
```
git clone https://github.com/ronichester/SL-animals-DVS-stbp
```
2. Download the dataset in [this link](http://www2.imse-cnm.csic.es/neuromorphs/index.php/SL-ANIMALS-DVS-Database);
3. Save the DVS recordings in the *data/recordings* folder and the file tags in the *data/tags* folder;
4. Run *slice_data.py* to slice the 59 raw recordings into 1121 samples.
```
python slice_data.py
```
5. Edit the custom parameters according to your preferences. The default parameters setting is functional and was tailored according to the information provided in the relevant papers, the reference codes used as a basis, and mostly by trial and error (lots of it!). You are encouraged to edit the main parameters in *sl_animals_stbp.py* and the arguments in the header of *layers.py*, and please **let me know if you got better results**.
6. Run *sl_animals_stbp.py* (or *train_test_only.py*) to start the SNN training:
```
python sl_animals_stbp.py
```
or
```
python train_test_only.py
```
7. The Tensorboard logs will be saved in *src/logs* and the network weights will be saved in *src/weights*. To visualize the logs with tensorboard:
  - open a terminal (I use Anaconda Prompt), go to the *src* directory and type:
```
tensorboard --logdir=logs
```
  - open your browser and type in the address bar http://localhost:6006/ or any other address shown in the terminal screen.
  

## References 
- Vasudevan, A., Negri, P., Di Ielsi, C. et al. ["SL-Animals-DVS: event-driven sign language animals dataset"](https://doi.org/10.1007/s10044-021-01011-w) . *Pattern Analysis and Applications 25, 505–520 (2021)*. 
- Yujie Wu, Lei Deng, Guoqi Li, Jun Zhu and Luping Shi: ["Spatio-Temporal Backprogation for Training High-performance Spiking Neural Networks"](https://www.frontiersin.org/articles/10.3389/fnins.2018.00331/full) *Frontiers in Neuroscience 12:331 (2018)* 
- The original dataset can be downloaded [here](http://www2.imse-cnm.csic.es/neuromorphs/index.php/SL-ANIMALS-DVS-Database)
- Other basic STBP implementations that served as a base for this project: [thiswinex/STBP-simple](https://github-com.translate.goog/thiswinex/STBP-simple?_x_tr_sl=auto&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=wapp) and [yjwu17/STBP-for-training-SpikingNN](https://github.com/yjwu17/STBP-for-training-SpikingNN#spatio-temporal-bp-for-spiking-neural-networks)

## Copyright
Copyright 2023 Schechter Roni. This software is free to use, copy, modify and distribute for personal, academic, or research use. Its terms are described under the General Public License, GNU v3.0.
