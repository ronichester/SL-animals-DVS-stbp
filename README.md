# SL-animals-DVS training with STBP
This repository contains an STBP (Spatio-Temporal Back Propagation) implementation on the SL-Animals-DVS dataset using Pytorch.

## Pre-requisites
- Pytorch

## Instructions
The SL-Animals-DVS dataset implementation code is in *dataset.py*, and it's basically a Pytorch Dataset object. The library [*Tonic*](https://tonic.readthedocs.io/en/latest/index.html#) was used to read and process the DVS recordings.

The core of the STBP method implementation is in *layers.py*: the base code code is from [thiswinex/STBP-simple](https://github-com.translate.goog/thiswinex/STBP-simple?_x_tr_sl=auto&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=wapp), to which I added a few fixes, changes and adaptations, inspired also by this other [STBP implementation](https://github-com.translate.goog/thiswinex/STBP-simple?_x_tr_sl=auto&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=wapp).

The Spiking Neural Network model is in *model.py* - **SLANIMALSNet()** - , and reproduces the architechture described in the SL-animals paper. The main simulation parameters are in the variable *args* located in the header of *layers.py*. Other custom parameters like 'batch size', 'data path', 'seed' and many others can be found in the beginning of *sl_animals_stbp.py* and changed according to your preferences.


## Use
1. Clone this repository:
```
git clone https://github.com/ronichester/SL-animals-DVS-stbp
```
2. Download the dataset in [this link](http://www2.imse-cnm.csic.es/neuromorphs/index.php/SL-ANIMALS-DVS-Database);
3. Save the DVS recordings in the *recordings* folder and the file tags in the *tags* folder;
4. The default parameters setting is functional and was tailored according to the information provided in the relevant papers, the reference codes used as a basis, and mostly by trial and error (lots of it!). You are encouraged to edit the main parameters in *sl_animals_stbp.py* and the arguments in the header of *layers.py* according to your personal choices, and let me know if you got better results.
5. Run *sl_animals_stbp.py* to start the SNN training:
```
python sl_animals_stbp.py
```
6. The Tensorboard logs will be saved in *src/logs* and the network weights will be saved in *src/weights*.

## References 
- Vasudevan, A., Negri, P., Di Ielsi, C. et al. ["SL-Animals-DVS: event-driven sign language animals dataset"](https://doi.org/10.1007/s10044-021-01011-w) . *Pattern Analysis and Applications 25, 505–520 (2021)*. 
- Yujie Wu, Lei Deng, Guoqi Li, Jun Zhu and Luping Shi: ["Spatio-Temporal Backprogation for Training High-performance Spiking Neural Networks"](https://www.frontiersin.org/articles/10.3389/fnins.2018.00331/full) *Frontiers in Neuroscience 12:331 (2018)* 
- The original dataset can be downloaded [here](http://www2.imse-cnm.csic.es/neuromorphs/index.php/SL-ANIMALS-DVS-Database)
- Other basic STBP implementations that served as a base for this project: [thiswinex/STBP-simple](https://github-com.translate.goog/thiswinex/STBP-simple?_x_tr_sl=auto&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=wapp) and [yjwu17/STBP-for-training-SpikingNN](https://github.com/yjwu17/STBP-for-training-SpikingNN#spatio-temporal-bp-for-spiking-neural-networks)
