# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:47:50 2023

Training the SL-Animals-DVS dataset with STBP: Spatio-temporal Back Propagation

Assumes the original dataset is already sliced in 1121 samples.
(Make sure to run "slice_data.py" before training)

@author: Schechter
"""
import os
import gc
import torch
import random
import numpy as np
import torch.optim as optim
from tensorboardX import SummaryWriter
from datetime import datetime
#imports from other python files
from model import SLANIMALSNet, SLANIMALSNet2
from dataset import AnimalsDvsSliced
from stbp_tools import kfold_split, train_net, get_params, get_optimizer

#assert we are on the right working directory (good for running on containers)
PATH = os.path.dirname(os.path.realpath(__file__))
os.chdir(PATH)

#load main parameters
net_params = get_params('network.yaml')

#initializing common arguments and use of GPU
use_cuda = net_params['Training']['cuda'] and torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}  #=1
seed = net_params['Training']['seed']       #fixing the seed
test_losses, test_accuracies = [], []       #initializing test history

#set the seed for reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

#creating a generator to split the data into 4 folds of train/test files
train_test_generator = kfold_split(net_params['Path']['file_list'],
                                   net_params['Training']['seed'])

#----------------------------- MAIN PROGRAM ------------------------------
#main program
if __name__ == '__main__':
    
    #print header
    print('WELCOME TO STBP TRAINING!')
    print("Training params: batch size={}, epochs={}, initial LR={}, binning mode={}"
          .format(net_params['Training']['batch'], 
                  net_params['Training']['epochs'], 
                  net_params['Training']['lr'], 
                  net_params['Training']['bin_mode']))
    print("Simulation params: steps={}, dt={}, Vth={}, Tau={}, a1={}".format(
          net_params['Simulation']['steps'],
          net_params['Simulation']['dt'], 
          net_params['Simulation']['Vth'], 
          net_params['Simulation']['tau'], 
          net_params['Simulation']['a1']))
    if net_params['Training']['random_crop'] :
        print('\nTraining with random crops... Testing steps = {}'.format(
            net_params['Testing']['steps']))
        print('    Training sample: {} ms  |  Testing sample: {} ms'.format(
            net_params['Simulation']['steps'] * net_params['Simulation']['dt'],
            net_params['Testing']['steps'] * net_params['Simulation']['dt']))
    print('\nStarting 4-fold cross validation (train/test only): Please wait...\n')
    global_st_time = datetime.now()       #monitor total training time 
    
    #CROSS-VALIDATION: iterate for each fold
    for fold, (train_set, test_set) in enumerate(train_test_generator, start=1):
        
        #logging statistics with tensorboard
        writer = SummaryWriter(
            net_params['Path']['logs'] + 'fold{}'.format(fold))
        
        #definining train and test Datasets
        training_set = AnimalsDvsSliced(
            dataPath     = net_params['Path']['data'],
            fileList     = train_set,
            samplingTime = net_params['Simulation']['dt'],
            timeSteps    = net_params['Simulation']['steps'], 
            randomCrop   = net_params['Training']['random_crop'],
            binMode      = net_params['Training']['bin_mode']
        )
        testing_set = AnimalsDvsSliced(
            dataPath     = net_params['Path']['data'],
            fileList     = test_set,
            samplingTime = net_params['Simulation']['dt'],
            timeSteps    = net_params['Testing']['steps'] if 
                           net_params['Training']['random_crop'] else 
                           net_params['Simulation']['steps'], 
            randomCrop   = False,  #always False for testing
            binMode      = net_params['Training']['bin_mode']
        )
        
        #definining train and test DataLoaders
        """
        If each sample on the dataset has the same length, batch_size can 
        be of any size (as long as it fits in the GPU memory). Since
        Animals-DVS dataset has different sample lengths, the only way to
        achieve training in batches is to crop all the samples to a fixed
        length size - STBP uses the first 1500 ms of all samples.
        """
        train_loader = torch.utils.data.DataLoader(
            dataset=training_set, 
            batch_size=net_params['Training']['batch'],
            shuffle=False, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            dataset=testing_set, 
            batch_size=net_params['Training']['tst_batch'],
            shuffle=False, **kwargs)
        
        #Defining the model
        model = SLANIMALSNet().to(device)
        
        #Defining the optimizer
        optimizer = get_optimizer(model, net_params)

        #Training and testing along the epochs
        print("FOLD {}:".format(fold))
        print("-----------------------------------------------")
        min_loss, max_acc = train_net(
            model, optimizer, device, train_loader, test_loader, 
            net_params['Training']['epochs'], 
            writer, fold, 
            net_params['Path']['weights'], 
            net_params['Training']['load_model'], 
            )
        
        #save this fold's losses and accuracies in history
        test_losses.append(min_loss)
        test_accuracies.append(max_acc)

        writer.close()
        #end of fold
        
        #clear memory
        gc.collect()
        del model
        torch.cuda.empty_cache()
    
    #end of cross-validation ---------------------------------------
    global_end_time = datetime.now()     #monitor total training time
    print('\nGlobal Training Time:', global_end_time - global_st_time)
    
    #print final results
    print("\nMin Test Loss on 4 folds:", test_losses)
    print("Min Test Loss:     {:.2f} +- {:.2f}".format(
        np.mean(test_losses), np.std(test_losses)))

    print("\nMax Test Accuracy on 4 folds:", test_accuracies)
    print("Max Test Accuracy:     {:.2f}% +- {:.2f}%".format(
        np.mean(test_accuracies), np.std(test_accuracies)))
    
        