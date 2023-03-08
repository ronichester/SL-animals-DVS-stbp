# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 17:39:43 2022

@author: Schechter
"""
#import libraries
import os
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import tonic.transforms as transforms
from sklearn.model_selection import KFold


#visualize dataset sample on animation
def animate_events(dataset, sample_index, time_window=50, frame_rate=24, repeat=False):
    """
    Generates an animation on a dataset sample. The sample is retrieved
    as a 1D array of events in the Tonic format (x, y, t, p), in [us] 
    resolution.
    
        dataset: torch Dataset object
        sample_index: int, must be between [0, 1120]
        time_window: int, time window in ms for each frame (default 50 ms)
        frame_rate: int, (default 24 FPS)
        repeat: bool, loop the animation infinetely (default is False)
    """
    #get sample events, class name, class index (ci) and sensor shape (ss)
    sample_events, sample_class, ci, ss = dataset.get_sample(sample_index)
       
    #create transform object
    sensor_size = (ss[0], ss[1], 2)                     #DVS sensor size
    frame_transform = transforms.Compose([
        transforms.Downsample(time_factor=0.001),       # us to ms
        transforms.TimeAlignment(),                     # 1st event at t=0
        transforms.ToFrame(sensor_size,                 # bin into frames
                           time_window=time_window)     # in [ms]
        ])  
    
    #transform event array -> frames (shape TCWH (time_bins, 2, 128, 128))
    frames = frame_transform(sample_events)
    
    #interval between frames in ms (default=41.6)
    interval = 1e3 / frame_rate  # in ms
    
    #defining 1st frame: image is the difference between polarities
    fig = plt.figure()
    plt.title("class name: {}".format(sample_class))
    image = plt.imshow((frames[0][1]-frames[0][0]).T, cmap='gray')
    
    #update the data on each frame
    def animate(frame):
        image.set_data((frame[1]-frame[0]).T)  
        return image

    animation = anim.FuncAnimation(fig, animate, frames=frames, 
                                   interval=interval, repeat=repeat)
    plt.show()
    
    return animation


#visualize dataset sample on plots (by time bins)
def plot_events(dataset, sample_index):
    """
    Generates a plot with 3 frames on a dataset sample. The events of a sample
    are divided in 3 time bins, each frame accumulates the events of 1 bin.
    """
    #get sample events, class name, class index (ci) and sensor shape (ss)
    sample_events, sample_class, ci, ss = dataset.get_sample(sample_index)
    
    #transform event array -> frames (shape TCWH)
    sensor_size = (ss[0], ss[1], 2)                     #DVS sensor size
    frame_transform = transforms.Compose([
        transforms.Downsample(time_factor=0.001),       # us to ms
        transforms.TimeAlignment(),                     # 1st event at t=0
        transforms.ToFrame(sensor_size, n_time_bins=3)  # events -> 3 frames
        ])  
    frames = frame_transform(sample_events)

    def plot_frames(frames):
        fig, axes = plt.subplots(1, len(frames))
        fig.suptitle("class name: {}".format(sample_class))
        for axis, frame in zip(axes, frames):
            axis.imshow((frame[1] - frame[0]).T, cmap='gray')
            axis.axis("off")
        plt.tight_layout()
        plt.show()
    
    plot_frames(frames)
    
    return frames


def kfold_split(fileList, seed, export_txt=False):
    """
    Split a file list (txt file) in 4 folds for cross validation (75%, 25%).
    It shuffles the files and then returns 4 separate training and test lists. 
    Optionally export the lists as txt files (default=False).
    
    Returns a generator.
    """
    def gen():  
        #load the files from .txt to an numpy 1D array
        files = np.loadtxt(fileList, dtype='str')  #[max 59 recordings]
        #create KFold object
        kf = KFold(n_splits=4, shuffle=True, random_state=seed)
        #create the folds
        for i, (train_index, test_index) in enumerate(kf.split(files), start=1):
            train_set, test_set = files[train_index], files[test_index]
            if export_txt:
                np.savetxt('../data/trainlist{}.txt'.format(i), train_set, 
                           fmt='%s')
                np.savetxt('../data/testlist{}.txt'.format(i), test_set, 
                           fmt='%s')
            yield train_set, test_set
    return gen()  #returns a generator!


def list_sliced_files(raw_file_list):
    #create a list of sliced files, given a list of 'raw' recording files
    sliced_file_list = []
    for file in raw_file_list:
        for i in range(19):
            sliced_file_list.append(file + '_{}.npy'.format(str(i).zfill(2)))
    
    return sliced_file_list


# Dacay learning_rate
def adjust_learning_rate(optimizer, epoch, lr_decay_epoch=25):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    print('Optimizer Learning Rate:', optimizer.param_groups[0]['lr'])
    return optimizer


#main training/evaluating function
def train_net(model, optimizer, device, train_loader, val_loader, epochs, 
              steps, writer, fold, save_path, pretrained, decay_lr=False):
    
    #load pre-trained weights
    if pretrained:
        print("Loading pre-trained weights...")
        checkpoint_path = save_path + 'model_weights_fold{}.pt'.format(fold)
        if os.path.isfile(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
            print('Model loaded.')
    
    #init best test results
    min_loss = 1.0
    max_acc = 0.0
    
    #main loop
    for epoch in range(1, epochs + 1):
        #training 
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.mse_loss(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #printing and logging training statistics (per batch)
            if (batch_idx % 10 == 0) or (
                    (batch_idx+1) % len(train_loader) == 0):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data / steps), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
                writer.add_scalar('Train Loss /batchidx', loss, 
                                  batch_idx + len(train_loader) * epoch)
        
        #evaluating
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += F.mse_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                #print(pred.shape)
                #print(target.shape)
                target_label = target.argmax(dim=1, keepdim=True)
                correct += pred.eq(target_label.view_as(pred)).sum().item()
        #avg test loss / accuracy
        val_loss /= len(val_loader.dataset)
        val_acc = 100. * correct / len(val_loader.dataset)
        
        #printing and logging testing statistics (per epoch)
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'
              .format(val_loss, correct, len(val_loader.dataset), val_acc))
        writer.add_scalar('Validation Loss /epoch', val_loss, epoch)
        writer.add_scalar('Validation Acc /epoch', val_acc, epoch)
        for i, (name, param) in enumerate(model.named_parameters()):
            if '_s' in name:
                writer.add_histogram(name, param, epoch)
        
        #updating max accuracy
        if val_acc > max_acc:
            max_acc = val_acc
        print('    (Min. Val. Loss: {:.4f}  |  Max. Val. Accuracy: {:.2f}%)\n'.format(min_loss, max_acc))

        #saving best weights for inference (at minimum loss)
        if val_loss < min_loss:
            min_loss = val_loss
            print('Lower minimum loss found!')
            print("Saving the model's best weights...\n")
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), 
                       save_path + "model_weights_fold{}.pt".format(fold))
            # torch.save(model, 
            #            save_path + "model_weights_fold{}.pth".format(fold))
        
        #adjusting learning rate (decay by 0.1 every 25 epochs)
        if decay_lr:
            optimizer = adjust_learning_rate(optimizer, epoch)
        #end of epoch
    
    #if model did not train for some reason, save the weights anyway for inference
    if min_loss == 1.0: 
        print("Unfortunately the model was not able to train on this fold with these settings.")
        print("Saving the model's weights...\n")
        os.makedirs(save_path, exist_ok=True)
        torch.save(model.state_dict(), 
                   save_path + "model_weights_fold{}.pt".format(fold))

    return min_loss, max_acc

def test_net(model, device, test_loader, writer, fold, save_path):
    
    #load best model weights for inference
    checkpoint_path = save_path + 'model_weights_fold{}.pt'.format(fold)
    if os.path.isfile(checkpoint_path):
        print("Loading pre-trained model...")
        model.load_state_dict(torch.load(checkpoint_path))
        print('Model loaded.')
    
    #testing loop
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            target_label = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(target_label.view_as(pred)).sum().item()
    #avg test loss / accuracy
    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    
    #printing and logging testing statistics
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'
            .format(test_loss, correct, len(test_loader.dataset), test_acc))
    writer.add_scalar('Test Loss', test_loss)
    writer.add_scalar('Test Accuracy', test_acc)

    return test_loss, test_acc


def stbp_init(weight_matrix):
    """
    Initializes a weight matrix with the method described in the STBP paper.
    The weigths are first sampled from a uniform distribution in [-1, 1], then:
        wij = wij / sqrt(summation(wij**2)) on the pre-synaptic neurons.
    """
    torch.nn.init.uniform_(weight_matrix, -1.0, 1.0)
    denominator = (weight_matrix**2).sum(dim=1).sqrt().unsqueeze(1)
    return weight_matrix / denominator
