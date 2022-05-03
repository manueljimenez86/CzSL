# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 11:21:14 2018

@author: psxmaji
"""

import pandas as pd
import numpy as np
import os
import sys
#import time
#from datetime import datetime
#import pdb

from _utils_CNN import *

from Convolutional_CNN import Convolutional_CNN

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

####################################################################################################
####################################################################################################

script_name = sys.argv[0] 
CWD = "/home/psxmaji" 
images_root = '__IMAGES__'
csv_root    = '__CSV__'

####################################################################################################

trial_name = os.path.basename(script_name)[:-3]

image_tuple   = (64, 64, 3)
images_folder = 'GZ1_Expert-ES_64x_tiff' 
images_csv = 'GZ1_Expert-ESRL.csv'

n_epochs = 100
n_executions = 10

####################################################################################################
#===================================================================================================
print_header(script_name)
#===================================================================================================
output_folder_root = directory_check(CWD, trial_name, preserve=False)

# Images and Labels loading:
images_folder = os.path.join(CWD, images_root, images_folder)
images_csv    = os.path.join(CWD, csv_root, images_csv)
images_labels = pd.read_csv(images_csv)

images_IDs = np.array(images_labels['OBJID'], dtype=str)

images_array = read_images_tensor(images_IDs, images_folder, image_tuple) 

partitions = [i for i in range(1, 6)]
executions = [j for j in range(1, n_executions + 1)]

for execution in executions :
    
    print('\n\n >> Running execution ', execution, ' <<<\n')
                    
    output_folder = directory_check(output_folder_root, trial_name + '_' + str(execution), preserve=False)
        
    
    for test_partition in partitions : 
        
        # Training, Validation and Test partitions:
        (images_train, labels_train, labels_train_vector, images_test, labels_test,
         labels_test_vector) = data_split_CNN_test(images_array, images_labels, 'ERL_EXPERT', 
                                                test_partition, n_splits=5, n_class=3)
        
        (CNN_train_images, CNN_train_labels, CNN_val_images,
         CNN_val_labels) = data_split_CNN_training(images_train, labels_train_vector, train_val_ratio=0.7)
        
        print('\n\n  >> ' + trial_name)
        print('\n  -> Partition ' + str(test_partition) + '\n')
        
        # CNN setting and training:
        CNN = Convolutional_CNN(image_tuple, n_class=3, model_weights=None)
        CNN.train(CNN_train_images, CNN_train_labels, CNN_val_images, CNN_val_labels, epochs=n_epochs)
        
        # Output log file:
        train_time = CNN.trial_log(output_folder, trial_name, test_partition=test_partition)
                
        # Test predictions and classification results: 
        predictions = CNN.model.predict(images_test)
        
        classification_performance_CNN(images_train, predictions, labels_test_vector, train_time,
                                    output_folder, trial_name, test_partition, n_class=3)
        
        save_predictions_ERL(labels_test, predictions, test_partition, output_folder, trial_name)
 
       
global_statistics_executions(CWD, trial_name, n_executions)
save_script(CWD, script_name, output_folder_root)
#===================================================================================================
print_header(script_name, end=True)
#===================================================================================================
