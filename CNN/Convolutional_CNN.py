# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 13:30:33 2019

@author: psxmaji
"""
import os
import sys
import numpy as np
import pandas as pd
import time
from datetime import datetime
import h5py

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg') 
plt.switch_backend('agg')
plt.rcParams.update({'figure.max_open_warning': 0})

from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Model
from keras import backend as K
from keras.applications import ResNet50

K.clear_session()

# Function to convert seconds into dd:hh:mm:ss
def convert_seconds (seconds) : 
    
    time = float(seconds)
    
    days = int(time // (24 * 3600))
    time = time % (24 * 3600)
    
    hours = str(int(time // 3600))      
    time %= 3600
    
    minutes = str(int(time // 60))
    time %= 60
    
    seconds = str(round(time, 2))
    
    if days != 0 : result = '(' + str(days) + 'd' + hours + 'h' + minutes + 'm' + seconds + 's)'
    
    else : result = '(' + hours + 'h' + minutes + 'm' + seconds + 's)' 
    
    return result

#==================================================================================================
#================================ Convolutional Neural Network ====================================
#==================================================================================================        
""" 
Convolutional Neural Network (CNN) made up of four consecutive Conv--Pooling pairs of layers. 
ReLU activation functions in all layers except the last one, which implements Sigmoid.
MaxPooling used. 

""" 

class Convolutional_CNN : 
    
    def __init__ (self, input_tuple, n_class=2, model_weights=None) : 
        
        self.input_tuple   = input_tuple
        self.n_class       = n_class
        self.model_weights = model_weights
        
        K.clear_session()
        
        self.build()
 

    def build (self) : 
        
        input_img = Input(shape=self.input_tuple) # (64, 64, 1)
        
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img) # (64, 64, 8)
        
        x = MaxPooling2D((2, 2), padding='same')(x)                          # (32, 32, 8)
        
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)          # (16, 16, 32)

        x = MaxPooling2D((2, 2), padding='same')(x)                          # (8, 8, 32)
        
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)          # (8, 8, 64)
        
        x = MaxPooling2D((2, 2), padding='same')(x)                    # (2, 2, 64) = 256 feats.
        
        encoded = Flatten()(x)
        
        x = Dense(256, activation='relu')(encoded)  # (2, 2, 64)
                
        x = Dense(128, activation='relu')(x)  # (8, 8, 32)
        
        output = Dense(self.n_class, activation='softmax')(x)
        
        self.model = Model(inputs=input_img, outputs=output)
        
        self.encoder = Model(input_img, encoded)
        
        if self.model_weights : self.model.load_weights(self.model_weights)
            
    
    def train (self, train_images, train_labels, val_images, val_labels, optimizer='sgd',
               loss='mean_squared_error', epochs=100, batch_size=256, shuffle=True) : 
        
        self.optimizer = optimizer
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.train_images = train_images
        self.train_labels = train_labels
        
        self.val_images = val_images
        self.val_labels = val_labels
        
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        
        start_time = time.time()
        record = self.model.fit(self.train_images, self.train_labels, epochs=self.epochs, shuffle=self.shuffle,
                                 validation_data=(self.val_images, self.val_labels))
        end_time = time.time() 

        self.history = record.history.copy()
        self.exec_time = end_time - start_time
        
    
    def trial_log (self, output_folder_path, trial_name, test_partition=None, save_weights=False) :
                
        self.output_folder_path = output_folder_path
        self.trial_name = trial_name
        self.test_partition = test_partition
        
        if self.test_partition : 
            
            output_path = os.path.join(self.output_folder_path, self.trial_name + '_' + str(self.test_partition))
        
        else: output_path = os.path.join(self.output_folder_path, self.trial_name)        
        
        runtime_str = convert_seconds(self.exec_time)
        train_time  = round(self.exec_time, 2)
       
        # History CSV:
        result_history = pd.DataFrame(data=self.history)
        history_csv_path = output_path + '_history.csv'
        result_history.to_csv(history_csv_path, index=None, float_format='%.4f')
       
        # History plot:
        plt.ioff()
        
        n = len(self.history['loss'])
        x = np.arange(1, n + 1, 1)
        plt.plot(x, self.history['loss'], label='loss')
        plt.plot(x, self.history['val_loss'], label='val_loss')
        
        step = int(n / 10) + 1
            
        plt.xticks([i for i in x], [str(i) if i % step == 0 else '' for i in x])
        plt.title(self.trial_name, fontsize=12)
        plt.legend()
        plt.grid()
        
        figure_path = output_path + '_training.png'
        plt.savefig(figure_path)
                
        # Log .txt file:        
        text_file = output_path + '_log.txt'
        
        file = open(text_file, 'w')

        file.write('\nOutput file generated from: ' + str(sys.argv[0]))       
        now = datetime.now()
        file.write('\r\nDate and time: ' + now.strftime("%x") + ' ' + now.strftime("%X")) 
        file.write('\r\n')        
        file.write('\r\nModel: Convolutional NN')
        file.write('\r\nInput images: ' + str(self.input_tuple))
        file.write('\r\n')
        file.write('\r\nOptimizer: ' + str(self.optimizer))
        file.write('\r\nLoss func.: ' + str(self.loss))
        file.write('\r\nEpochs: ' + str(self.epochs))
        file.write('\r\nBatch size: ' + str(self.batch_size))
        file.write('\r\nShuffle: ' + str(self.shuffle))
        file.write('\r\n')
        file.write('\r\nTrain      : ' + str(len(self.train_images)) + ' examples')
        file.write('\r\nValidation : ' + str(len(self.val_images)) + ' examples')
        file.write('\r\nTotal: ' + str(len(self.train_images) + len(self.val_images)) + ' examples')
        file.write('\r\n')
        file.write('\r\nExecution time: ' + str(round(self.exec_time, 2)) + ' ' + runtime_str)
        file.write('\r\n')
        file.write('\r\nCNN structure summary:\n')
        file.write('\r\n')
        
        self.model.summary(print_fn=lambda x : file.write(x + '\r\n'))
        
        file.close()
        
        if save_weights : 
            
            filepath = output_path + '_weights.hdf5'
            self.model.save_weights(filepath)
            
            return filepath, train_time
                
        else: return train_time
        
                
###################################################################################################        
###################################################################################################        
###################################################################################################