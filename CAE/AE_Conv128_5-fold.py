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
from datetime import datetime
#import pdb

from _utils_AE import *
from Conv128_AE import Basic_Conv128_Autoencoder

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

###################################################################################################
###################################################################################################
image_size     = int(sys.argv[1])
test_partition = int(sys.argv[2])
script_name = sys.argv[0] 
CWD = "/home/psxmaji" 
images_root = "__IMAGES__"
csv_root    = "__CSV__"
###################################################################################################
trial_name = 'Conv128_GZ1-T2_RGB-' + str(image_size) + 'x-tiff'

image_tuple   = (image_size, image_size, 3)
images_folder = 'GZ1_T2_' + str(image_size) + 'x_tiff' 
images_csv = "GZ1_T2_Votes-Scores-Flags.csv"
###################################################################################################
#==================================================================================================
print_header(script_name)
#==================================================================================================
output_folder   = directory_check(CWD, trial_name, preserve=False)

# Images and Labels loading:
images_folder = os.path.join(CWD, images_root, images_folder)
images_csv    = os.path.join(CWD, csv_root, images_csv)
images_labels = pd.read_csv(images_csv)

images_IDs    = np.array(images_labels['OBJID'], dtype=str)

# Training, Validation and Test partitions:
(labels_train, labels_test) = labels_split_test(images_labels, test_partition)

images_train_IDs   = np.array(labels_train['OBJID'], dtype=str)
images_array_train = read_images_tensor_by_ID(images_IDs, images_train_IDs, images_folder,
                                               image_tuple, file_ext='.tiff') 

(AE_train, AE_validation) = data_split_AE_training(images_array_train, train_val_ratio=0.7)

print('\n\n  >> Running ' + trial_name)
print('\n  -> Partition ' + str(test_partition))
print_time()

# Autoencoder setting and training:
autoencoder = Basic_Conv128_Autoencoder(image_tuple)
autoencoder.train(AE_train, AE_validation, epochs=100)

# Images encoding and feats. saving:
encoded_images_train = autoencoder.encoder.predict(images_array_train) 
save_features_csv(encoded_images_train, labels_train, output_folder,
              concat_str([trial_name, 'train', test_partition]))

del(images_array_train)

images_test_IDs   = np.array(labels_test['OBJID'], dtype=str)
images_array_test = read_images_tensor_by_ID(images_IDs, images_test_IDs, images_folder,
                                               image_tuple, file_ext='.tiff') 

encoded_images_test  = autoencoder.encoder.predict(images_array_test)
save_features_csv(encoded_images_test, labels_test, output_folder,
              concat_str([trial_name, 'test', test_partition]))

# Output log file:
autoencoder.trial_log(output_folder, trial_name, test_partition=test_partition, csv_global=True)

#==================================================================================================
print_header(script_name, end=True)
#==================================================================================================
