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

from sklearn.neural_network import MLPClassifier

from _utils_CNN import *

from Convolutional_CNN import Convolutional_CNN

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

####################################################################################################
####################################################################################################

data_partition = sys.argv[2]
CWD            = sys.argv[1] 
script_name    = sys.argv[0] 

#CWD = "/home/psxmaji" 
images_root = '__IMAGES__'
csv_root    = '__CSV__'

####################################################################################################

trial_name = os.path.basename(script_name)[:-3]

image_tuple       = (64, 64, 3)
images_folder     = 'GZ1_Expert-MG_64x_tiff' 
images_subfolder  = 'GZ1_Expert-MG_partitions'
AE_weights_folder = 'CAE_weights_MG' 

images_csv_train  = 'GZ1_Expert-MG_train_' + data_partition + '.csv'
images_csv_test   = 'GZ1_Expert-MG_test_' + data_partition + '.csv' 
images_csv_master = 'GZ1_Expert-MG.csv' 

n_epochs     = 100
n_executions = 10

####################################################################################################
#===================================================================================================
print_header(script_name)
#===================================================================================================
output_folder_root = directory_check(CWD, trial_name, preserve=False)

# Images and Labels loading:
images_folder = os.path.join(CWD, images_root, images_folder)

images_csv_train    = os.path.join(CWD, csv_root, images_subfolder, images_csv_train)
images_labels_train = pd.read_csv(images_csv_train)

images_csv_test    = os.path.join(CWD, csv_root, images_subfolder, images_csv_test)
images_labels_test = pd.read_csv(images_csv_test)

images_IDs_train = np.array(images_labels_train['OBJID'], dtype=str)
images_IDs_test  = np.array(images_labels_test['OBJID'], dtype=str)

images_csv_master = os.path.join(CWD, csv_root, images_csv_master)
images_csv_master = pd.read_csv(images_csv_master)
images_IDs_master = np.array(images_csv_master['OBJID'], dtype=str)

images_array_train = read_images_tensor_by_ID(images_IDs_master, images_IDs_train, images_folder,
                                              image_tuple)
images_array_test  = read_images_tensor_by_ID(images_IDs_master, images_IDs_test, images_folder,
                                              image_tuple)

# Images and labels split for TR and TS: 
split_1 = int(0.05 * len(images_array_train))
split_2 = split_1 + int(0.20 * len(images_array_train))

images_L  = images_array_train[0 : split_1]
images_LA = images_array_train[0 : split_2] 
images_U  = images_array_train[split_1 :]

labels_L = images_labels_train[0 : split_1]
labels_L = labels_L.reset_index()
labels_L = pd.DataFrame(data=labels_L, columns=list(images_labels_train.columns))

labels_A = images_labels_train[split_1 : split_2]
labels_A = labels_A.reset_index()
labels_A = pd.DataFrame(data=labels_A, columns=list(images_labels_train.columns))

labels_LA = images_labels_train[0 : split_2]
labels_LA = labels_LA.reset_index()
labels_LA = pd.DataFrame(data=labels_LA, columns=list(images_labels_train.columns))

labels_U = images_labels_train[split_1 :]
labels_U = labels_U.reset_index()
labels_U = pd.DataFrame(data=labels_U, columns=list(images_labels_train.columns))

labels_L_vector  = vectorize_labels_bin(np.array(labels_L['MG_EXPERT'], dtype=int))
labels_LA_vector = vectorize_labels_bin(np.array(labels_LA['MG_AMATEUR'], dtype=int))

# Ground-truth Expert labels:
labels_L_bin    = np.array(labels_L['MG_EXPERT'], dtype=int)
labels_U_bin    = np.array(labels_U['MG_EXPERT'], dtype=int)
labels_test_bin = np.array(images_labels_test['MG_EXPERT'], dtype=int)
labels          = (labels_L_bin, labels_U_bin, labels_test_bin) 

tag_pretraining = 'CzSL_Pre-training-RUS_Expert-MG'
tag_finetuning  = 'CzSL_Fine-tuning-RUS_Expert-MG'

executions = [j for j in range(1, n_executions + 1)]

for execution in executions : 

    print('\n\n  >> Amateur pre-training...')
    print('\n  -> Running partition ' + data_partition + ' and execution ' + str(execution) + '\n')
    
    output_folder = directory_check(output_folder_root, trial_name + '_' + str(execution),
                                    preserve=False)
    
    # (1.0) ROS/RUS application:
    (images_train, labels_train_vector) = Random_Under_Over_Sampling(images_LA, labels_LA_vector,
                                                                     oversampling=False)
    
    ratio = imbalance_ratio(labels_train_vector)
    
    # (1.1) Pre-training Amateur:
    (CNN_train_images, CNN_train_labels, CNN_val_images,
     CNN_val_labels) = data_split_CNN_training(images_train, labels_train_vector, train_val_ratio=0.7) 
    
    AE_weights_filename = 'CAE_Expert-MG_FE_' + data_partition + '_encoder_weights.hdf5' 
    AE_weights_filename = os.path.join(CWD, AE_weights_folder, AE_weights_filename)
    
    CNN_pretraining = Convolutional_CNN(image_tuple, n_class=2, model_weights=AE_weights_filename)
    
    CNN_pretraining.train(CNN_train_images, CNN_train_labels, CNN_val_images, CNN_val_labels,
                          epochs=n_epochs)
    weights_filepath, train_time = CNN_pretraining.trial_log(output_folder, tag_pretraining,
                                                             test_partition=data_partition,
                                                             save_weights=True)
    
    # Pre-training predictions: 
    predictions_L    = CNN_pretraining.model.predict(images_L)
    predictions_U    = CNN_pretraining.model.predict(images_U)
    predictions_test = CNN_pretraining.model.predict(images_array_test)
    
    predictions = [predictions_L, predictions_U, predictions_test] 
    
    classification_performance_SSL_imbalance(images_train, predictions, labels, ratio, output_folder,
                                             tag_pretraining, train_time, int(data_partition))
    
    save_predictions_MG_SSL(labels_L, predictions[0], data_partition, 'L', output_folder, tag_pretraining)
    save_predictions_MG_SSL(labels_U, predictions[1], data_partition, 'U', output_folder, tag_pretraining)
    save_predictions_MG_SSL(images_labels_test, predictions[2], data_partition, 'test', output_folder,
                         tag_pretraining)
    
    print('\n\n  >> Expert fine-tuning...')
    print('\n  -> Running partition ' + data_partition + '\n')
    
    # MLP knowledge transfer: 
    config = config = (['EL', 'CS', 'CW', 'ACW', 'EDGE', 'DK', 'MG'], (8, 7, 5, 3))
    
    MLP_scores_L = np.array(labels_L[config[0]], dtype=float)
    MLP_scores_A = np.array(labels_A[config[0]], dtype=float)
    MLP_scores_U = np.array(labels_U[config[0]], dtype=float)
    
    MLP = MLPClassifier(hidden_layer_sizes=config[1], solver='adam', alpha=1e-4, random_state=0,
                        max_iter=300)
    
    MLP.fit(MLP_scores_L, labels_L_vector)
    
    MLP_predictions = check_bin_labels(MLP.predict(MLP_scores_A))
    labels_MLP_vector = np.vstack((labels_L_vector, MLP_predictions)) 
    
    labels_U_MLP = MLP.predict_proba(MLP_scores_U)
    labels_U['MLP_NON-MG'] = labels_U_MLP[:, 0]
    labels_U['MLP_MG'] = labels_U_MLP[:, 1] 
    
    # (2.0) ROS/RUS application:
    (images_train, labels_train_vector) = Random_Under_Over_Sampling(images_LA, labels_MLP_vector,
                                                                     oversampling=False)
    
    ratio = imbalance_ratio(labels_train_vector)

    # (2.1) Fine-tuning Expert:
    (CNN_train_images, CNN_train_labels, CNN_val_images,
     CNN_val_labels) = data_split_CNN_training(images_train, labels_train_vector, train_val_ratio=0.7) 
    
    CNN_finetuning = Convolutional_CNN(image_tuple, n_class=2, model_weights=weights_filepath)
    
    CNN_finetuning.train(CNN_train_images, CNN_train_labels, CNN_val_images, CNN_val_labels,
                         epochs=n_epochs)
    weights_filepath, train_time = CNN_finetuning.trial_log(output_folder, tag_finetuning,
                                                             test_partition=data_partition,
                                                             save_weights=True)
    
    # Final predictions: 
    predictions_L    = CNN_finetuning.model.predict(images_L)
    predictions_U    = CNN_finetuning.model.predict(images_U)
    predictions_test = CNN_finetuning.model.predict(images_array_test)
    
    predictions = [predictions_L, predictions_U, predictions_test] 
    
    classification_performance_SSL_imbalance(images_train, predictions, labels, ratio, output_folder,
                                             tag_finetuning, train_time, int(data_partition))
    
    save_predictions_MG_SSL(labels_L, predictions[0], data_partition, 'L', output_folder, tag_finetuning)
    save_predictions_MG_SSL(labels_U, predictions[1], data_partition, 'U', output_folder, tag_finetuning)
    save_predictions_MG_SSL(images_labels_test, predictions[2], data_partition, 'test', output_folder,
                         tag_finetuning) 

if data_partition == '5' : global_statistics_executions_SSL_imbalance(CWD, tag_finetuning, trial_name,
                                                                      n_executions)
#===================================================================================================
print_header(script_name, end=True)
#===================================================================================================
