# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:47:50 2023

Training the SL-Animals-DVS dataset with STBP: Spatio-temporal Back Propagation

Assumes the original dataset is already sliced in 1121 samples.
(Make sure to run "slice_data.py" before training)

@author: Schechter
"""
import os
import torch
import random
import numpy as np
import torch.optim as optim
from tensorboardX import SummaryWriter
from datetime import datetime
from sklearn.model_selection import train_test_split
#imports from other python files
from model import SLANIMALSNet, SLANIMALSNet2
from dataset import AnimalsDvsSliced
from stbp_tools import kfold_split, train_net, test_net, get_params, get_optimizer

#assert we are on the right working directory (good for running on containers)
PATH = os.path.dirname(os.path.realpath(__file__))
os.chdir(PATH)

#load main parameters
net_params = get_params('network.yaml')

#initializing common arguments and use of GPU
use_cuda = net_params['Training']['cuda'] and torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}  #=1
seed = net_params['Training']['seed']     #fixing the seed
val_losses, val_accuracies = [], []       #initializing val history
test_losses, test_accuracies = [], []     #initializing test history

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
    print('\nStarting 4-fold cross validation (train/validation/test): Please wait...\n')
    global_st_time = datetime.now()       #monitor total training time 
    
    #CROSS-VALIDATION: iterate for each fold
    for fold, (train_set, test_set) in enumerate(train_test_generator, start=1):
        
        #logging statistics with tensorboard
        writer = SummaryWriter(
            net_params['Path']['logs'] + 'fold{}'.format(fold))

        #divide train_set into train and validation (85-15)
        train_set, val_set = train_test_split(
            train_set, 
            test_size=0.15, 
            random_state=net_params['Training']['seed'] + 1
            )
        
        #definining train and test Datasets
        training_set = AnimalsDvsSliced(
            dataPath     = net_params['Path']['data'],
            fileList     = train_set,
            samplingTime = net_params['Simulation']['dt'],
            sampleLength = net_params['Simulation']['dt'] 
                           * net_params['Simulation']['steps'],
            fixedLength  = True, 
            binMode      = net_params['Training']['bin_mode']
        )
        validation_set = AnimalsDvsSliced(
            dataPath     = net_params['Path']['data'],
            fileList     = val_set,
            samplingTime = net_params['Simulation']['dt'],
            sampleLength = net_params['Simulation']['dt']
                           * net_params['Simulation']['steps'],
            fixedLength  = True, 
            binMode      = net_params['Training']['bin_mode']
        )
        testing_set = AnimalsDvsSliced(
            dataPath     = net_params['Path']['data'],
            fileList     = test_set,
            samplingTime = net_params['Simulation']['dt'],
            sampleLength = net_params['Simulation']['dt'] 
                           * net_params['Simulation']['steps'],
            fixedLength  = True, 
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
        val_loader = torch.utils.data.DataLoader(
            dataset=validation_set, 
            batch_size=net_params['Training']['tst_batch'],
            shuffle=False, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            dataset=testing_set, 
            batch_size=net_params['Training']['tst_batch'],
            shuffle=False, **kwargs)     
   
        #Defining the model
        model = SLANIMALSNet().to(device)
        # model = SLANIMALSNet2().to(device)  #under development
        
        #Defining the optimizer
        optimizer = get_optimizer(model, net_params)
        
        #train the network
        print("TRAINING FOLD {}:".format(fold))
        print("-----------------------------------------------")
        min_loss, max_acc = train_net(
            model, optimizer, device, train_loader, val_loader, 
            net_params['Training']['epochs'], 
            net_params['Simulation']['steps'], 
            writer, fold, 
            net_params['Path']['weights'], 
            net_params['Training']['load_model'], 
            )
        
        #test the network
        print("TESTING FOLD {}:".format(fold))
        print("-----------------------------------------------")
        test_loss, test_acc = test_net(model, device, test_loader, writer, fold, 
                                       net_params['Path']['weights'])
        
        #save this fold's losses and accuracies in history
        val_losses.append(min_loss)
        val_accuracies.append(max_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        writer.close()
        #end of fold
    
    #end of cross-validation ---------------------------------------
    global_end_time = datetime.now()     #monitor total training time
    print('\nGlobal Training Time:', global_end_time - global_st_time)
    
    #print final results
    print("\nMin Val. Loss on 4 folds:", val_losses)
    print("Min Val. Loss:     {:.2f} +- {:.2f}".format(
        np.mean(val_losses), np.std(val_losses)))

    print("\nMax Val. Accuracy on 4 folds:", val_accuracies)
    print("Max Val. Accuracy:     {:.2f}% +- {:.2f}%".format(
        np.mean(val_accuracies), np.std(val_accuracies)))

    print("\nTest Loss on 4 folds:", test_losses)
    print("Average Test Loss:     {:.2f} +- {:.2f}".format(
        np.mean(test_losses), np.std(test_losses)))

    print("\nTest Accuracy on 4 folds:", test_accuracies)
    print("Average Test Accuracy:     {:.2f}% +- {:.2f}%".format(
        np.mean(test_accuracies), np.std(test_accuracies)))
    
        