# SL-animals-DVS training with STBP
This repository contains an STBP (Spatio-Temporal Back Propagation) implementation on the SL-Animals-DVS dataset using Pytorch.

## Pre-requisites
While not sure if the list below contain the actual minimums, it will run for sure if you do have the following:
- Python 3.0+
- Pytorch 1.11+
- CUDA 11.3+
- python libraries: os, numpy, matplotlib, pandas, sklearn, datetime, tonic, tensorboardX

## README FIRST
This package contains the necessary python files to train a Spiking Neural Network with the STBP method on the Sign Language Animals DVS dataset. 

**A BRIEF INTRODUCTION:**  
STBP is an offline training method that directly trains a SNN, expanding the use of the classic backpropagation algorithm to the time domain, so the training occurs in space AND time. Therefore, it is a suitable method to train SNNs, which are biologically plausible networks (in short).  
The SL-animals-DVS is a dataset of sign language (SL) gestures peformed by different people representing animals, and recorded with a Dynamic Vision Sensor (DVS).

**IMPLEMENTATION**  
Package Contents:  
- dataset.py
- layers.py
- model.py
- sl_animals_stbp.py
- slice_data.py
- stbp_tools.py

The SL-Animals-DVS dataset implementation code is in *dataset.py*, and it's basically a Pytorch Dataset object. The library [*Tonic*](https://tonic.readthedocs.io/en/latest/index.html#) was used to read and process the DVS recordings.

The core of the STBP method implementation is in *layers.py*: the base code code is from [thiswinex/STBP-simple](https://github-com.translate.goog/thiswinex/STBP-simple?_x_tr_sl=auto&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=wapp), to which I added a few fixes, changes and adaptations, inspired also by this other [STBP implementation](https://github-com.translate.goog/thiswinex/STBP-simple?_x_tr_sl=auto&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=wapp). The main simulation parameters are in the variable *args* located in the header of *layers.py*.

The Spiking Neural Network model is in *model.py* (SLANIMALSNet), and reproduces the architecture described in the SL-animals paper. The main program is in *sl_animals_stbp.py*, which contains right at the top main parameters that can be customized like 'batch size', 'data path', 'seed' and many others.  The main training tools and functions used in the package are in *stbp_tools.py*. 

Finally, *slice_data.py* is a script to slice the SL animals DVS recordings into actual samples for training, and save the slices to disk. The reason for that is because the original raw dataset after download contains only 59 files (DVS recordings), and not 1121 samples. Each recording contains 1 individual performing the 19 gestures in sequence, so there is a need to manually cut these 19 slices from each whole recording in order to actually use the dataset. 


## Use
1. Clone this repository:
```
git clone https://github.com/ronichester/SL-animals-DVS-stbp
```
2. Download the dataset in [this link](http://www2.imse-cnm.csic.es/neuromorphs/index.php/SL-ANIMALS-DVS-Database);
3. Save the DVS recordings in the *recordings* folder and the file tags in the *tags* folder;
4. Run *slice_data.py* to slice the 59 raw recordings into 1121 samples.
```
python slice_data.py
```
5. Edit the custom parameters according to your preferences. The default parameters setting is functional and was tailored according to the information provided in the relevant papers, the reference codes used as a basis, and mostly by trial and error (lots of it!). You are encouraged to edit the main parameters in *sl_animals_stbp.py* and the arguments in the header of *layers.py*, and please **let me know if you got better results**.
6. Run *sl_animals_stbp.py* to start the SNN training:
```
python sl_animals_stbp.py
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
Copyright 2023 Schechter Roni. This software is free to use, copy, modify and distribute for personal, academic, or research use. It is **not authorized for commercial use** without strict and written authorization by the author. License terms are described in Creative Commons License CC BY-NC.
