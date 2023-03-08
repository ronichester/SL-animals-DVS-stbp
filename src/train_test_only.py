# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:47:50 2023

Training the SL-Animals-DVS dataset with STBP: Spatio-temporal Back Propagation

Assumes the original dataset is already sliced in 1121 samples.
(Make sure to run "slice_data.py" before training)

@author: Schechter
"""
import os
import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from datetime import datetime
#imports from other python files
from model import SLANIMALSNet, SLANIMALSNet2
from dataset import AnimalsDvsSliced
from stbp_tools import kfold_split, train_net, Params

#assert we are on the right working directory (good for running on containers)
PATH = os.path.dirname(os.path.realpath(__file__))
os.chdir(PATH)

#load main parameters
net_params = Params('network.yaml')

#Setting the seed, use of GPU and initializing common arguments
torch.manual_seed(net_params.get_param('Training.seed'))
use_cuda = net_params.get_param('Training.cuda') and torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}  #=1
test_losses, test_accuracies = [], []       #initializing test history

#creating a generator to split the data into 4 folds of train/test files
train_test_generator = kfold_split(net_params.get_param('Path.file_list'),
                                   net_params.get_param('Training.seed'))

#----------------------------- MAIN PROGRAM ------------------------------
#main program
if __name__ == '__main__':
    
    #print header
    print('WELCOME TO STBP TRAINING!')
    print("Training params: batch size={}, epochs={}, initial LR={}, binning mode={}"
          .format(net_params.get_param('Training.batch'), 
                  net_params.get_param('Training.epochs'), 
                  net_params.get_param('Training.lr'), 
                  net_params.get_param('Training.bin_mode')))
    print("Simulation params: steps={}, dt={}, Vth={}, Tau={}, a1={}".format(
          net_params.get_param('Simulation.steps'),
          net_params.get_param('Simulation.dt'), 
          net_params.get_param('Simulation.Vth'), 
          net_params.get_param('Simulation.tau'), 
          net_params.get_param('Simulation.a1')))
    print('\nStarting 4-fold cross validation (train/test only): Please wait...\n')
    global_st_time = datetime.now()       #monitor total training time 
    
    #CROSS-VALIDATION: iterate for each fold
    for fold, (train_set, test_set) in enumerate(train_test_generator, start=1):
        
        #logging statistics with tensorboard
        writer = SummaryWriter(
            net_params.get_param('Path.logs') + 'fold{}'.format(fold))
        
        #definining train and test Datasets
        training_set = AnimalsDvsSliced(
            dataPath     = net_params.get_param('Path.data'),
            fileList     = train_set,
            samplingTime = net_params.get_param('Simulation.dt'),
            sampleLength = net_params.get_param('Simulation.dt') 
                           * net_params.get_param('Simulation.steps'),
            fixedLength  = True, 
            binMode      = net_params.get_param('Training.bin_mode')
        )
        testing_set = AnimalsDvsSliced(
            dataPath     = net_params.get_param('Path.data'),
            fileList     = test_set,
            samplingTime = net_params.get_param('Simulation.dt'),
            sampleLength = net_params.get_param('Simulation.dt') 
                           * net_params.get_param('Simulation.steps'),
            fixedLength  = True, 
            binMode      = net_params.get_param('Training.bin_mode')
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
            batch_size=net_params.get_param('Training.batch'),
            shuffle=False, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            dataset=testing_set, 
            batch_size=net_params.get_param('Training.tst_batch'),
            shuffle=False, **kwargs)
        
        #Defining the model
        model = SLANIMALSNet().to(device)
        # model = SLANIMALSNet2().to(device)  #under development
        
        #Defining the optimizer
        if net_params.get_param('Training.optimizer') == 'ADAM':
            optimizer = optim.Adam(model.parameters(), 
                                   lr=net_params.get_param('Training.lr'))
        elif net_params.get_param('Training.optimizer') == 'SGD':
            optimizer = optim.SGD(model.parameters(), 
                                  lr=net_params.get_param('Training.lr'),
                                  momentum=net_params.get_param('Training.momentum'))
        else:
            print("Optimizer option is not valid; using ADAM instead.")
            optimizer = optim.Adam(model.parameters(), 
                                   lr=net_params.get_param('Training.lr'))
        
        #Training and testing along the epochs
        print("FOLD {}:".format(fold))
        print("-----------------------------------------------")
        min_loss, max_acc = train_net(
            model, optimizer, device, train_loader, test_loader, 
            net_params.get_param('Training.epochs'), 
            net_params.get_param('Simulation.steps'), 
            writer, fold, 
            net_params.get_param('Path.weights'), 
            net_params.get_param('Training.load_model'), 
            )
        
        #save this fold's losses and accuracies in history
        test_losses.append(min_loss)
        test_accuracies.append(max_acc)

        writer.close()
        #end of fold
    
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
    
        