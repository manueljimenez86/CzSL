# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 17:24:51 2018

@author: psxmaji
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os
import sys
from datetime import datetime
import random
import shutil

from imblearn.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score

matplotlib.use('Agg') 
plt.switch_backend('agg')

# =============================================================================
# *********************** F U N C T I O N S ***********************************
# =============================================================================

# (1.1) Function to save in .csv format the autoencoder results  
def save_features_csv (encoded_imgs, images_labels, output_folder, trial_name) : 
    
    if len(encoded_imgs) != len(images_labels) : 
    
        print('\n >> ERROR _utils_AE(1.1): features tensor and labels array length do not match!!\n')
        sys.exit()
    
    
    if len(encoded_imgs.shape) > 2 :
        
        num_features = np.prod(encoded_imgs.shape[1:])
        num_imgs     = len(encoded_imgs)
        
        enc_imgs_flat = np.zeros((num_imgs, num_features), dtype=float)
        
        for i in range(0, num_imgs) : 
            
            enc_imgs_flat[i] = encoded_imgs[i].flatten()
            
    else : enc_imgs_flat = encoded_imgs
    
    features = pd.DataFrame(enc_imgs_flat, index=None)
    result = images_labels.join(features)
    
    result_path = os.path.join(output_folder, trial_name + '_feats.csv')
    result.to_csv(result_path, index=None, float_format='%.4g')

    return



# (2.1) Function to read greyscale images 
def read_images_flat (IDs_array, folder_path, dim_tuple, file_extension='.png', normalisation=True) :
    
    """
    This function reads the sets of 1,000 greyscale images present in sub-folders of 
    'parent folder', following the order in 'IDs_array'. 'dim_tuple' is the 
    tensorial dimensionality of the image in the format (width, height).
    (Channels is supposed to be equal to '1').
    """

    images_array = np.zeros((len(IDs_array), dim_tuple[0] * dim_tuple[1]), dtype=float)           
    folders = sorted([x for x in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, x))])

    print('\nLoading images...\n')

    for i in range(0, len(IDs_array)) : 
        
        if i % 1000 == 0 : 
            
            print(' -> ' + str(i) + ' images done...')
            
            folder = folders[int(i / 1000)]
        
        image_name = IDs_array[i] + file_extension
        image_path = os.path.join(folder_path, folder, image_name)
        
        images_array[i] = cv2.imread(image_path, 0).flatten()
        
    print('\n\nImages load successful!!\n')

    if normalisation == True : images_array = images_array.astype('float32') / 255.0
    
    return images_array



#(2.2) Function to read RGB images by folder
def read_images_tensor (IDs_array, folder_path, dim_tuple, file_extension='.tiff') :
    
    """
    This function reads the sets of 1,000 (RGB) images present in sub-folders of 
    'parent folder', following the order in 'IDs_array'. 'dim_tuple' is the 
    tensorial dimensionality of the image in the format (width, height, channels).
    """

    array_tuple = (len(IDs_array),) + dim_tuple
    RGB = 1
            
    images_array = np.zeros(array_tuple, dtype=float)
    
    folders = sorted([x for x in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, x))])
    
    if dim_tuple[-1] == 1 : RGB = 0

    print('\nLoading images...\n')
    
    for i in range(0, len(IDs_array)) : 
        
        if i % 1000 == 0 : 
            
            print(' -> ' + str(i) + ' images done...')
            
            folder = folders[int(i / 1000)]
        
        image_name = IDs_array[i] + file_extension
        image_path = os.path.join(folder_path, folder, image_name)
        
        image           = cv2.imread(image_path, RGB)
        image_reshaped  = image.reshape(dim_tuple)
        images_array[i] = image_reshaped.astype('float32') / 255.0
                
    print('\n\nImages load successful!\n')
          
    return images_array



#(2.3) Function to read RGB images by image ID
def read_images_tensor_by_ID (IDs_array, IDs_array_to_read, folder_path, dim_tuple, file_ext='.tiff') :
    
    """
    This function reads the sets of 1,000 (RGB) images present in sub-folders of 
    'parent folder', following the order in 'IDs_array'. 'dim_tuple' is the 
    tensorial dimensionality of the image in the format (width, height, channels).
    """

    array_tuple = (len(IDs_array_to_read),) + dim_tuple
    RGB = 1
    if dim_tuple[-1] == 1 : RGB = 0
            
    images_array = np.zeros(array_tuple, dtype=float)
    
    folders = sorted([x for x in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, x))])
    
    print('\nLoading images...\n')
    
    for i in range(0, len(IDs_array_to_read)) : 
        
        if i % 1000 == 0 and i != 0 : print(' -> ' + str(i) + ' images done...')
        
        image_index = np.where(IDs_array == IDs_array_to_read[i])
        image_index = image_index[0][0]
        
        image_folder = folders[int(image_index / 1000)]
                    
        image_name = IDs_array_to_read[i] + file_ext
        image_path = os.path.join(folder_path, image_folder, image_name)
        
        image = cv2.imread(image_path, RGB)
        image_reshaped = image.reshape(dim_tuple)
        images_array[i] = image_reshaped.astype('float32') / 255.0
                
    print('\n\nImages load successful!\n')
          
    return images_array



# (5) Function to concatenate words in file names
def concat_str (str_list, file_ext='', separator='_') : 
    
    result = str_list[0]
    
    for i in range(1, len(str_list)) : 
        
        if type(str_list[i]) != str : str_list[i] = str(str_list[i])
        
        if str_list[i] != '' : result = result + separator + str_list[i]
        
    if file_ext != '' : result = result + file_ext
    
    return result



# (6) Function to create a directory for output files
def directory_check (root_path, directory_name, preserve=True) :
    
    list_dir = os.listdir(root_path)
    directory_path = os.path.join(root_path, directory_name)
    
    if preserve == False :
        
        if directory_name not in list_dir : os.makedirs(directory_path)
            
    else: 
            
        done = False
        length = len(directory_name)
        n = 3
        
        while done == False : 
                  
            if directory_name in list_dir and len(directory_name) == length : 
                
                directory_name += '(2)'
                
            elif directory_name in list_dir :  
                
                index_parenthesis = directory_name.rindex('(') + 1
                
                directory_name = directory_name[0:index_parenthesis] + str(n) + ')'
                n += 1
                
            else : 
                
                directory_path = os.path.join(root_path, directory_name)
                os.makedirs(directory_path)
                done = True
                        
    return directory_path



# (9) Function to print informative headers to the screen while running the scripts
def print_header (script_name, end=False) : 
 
    now = datetime.now()

    if end == True : 
        
        print('\n\n=============================================================')
        print('\n>> Running of ' + str(script_name) + ' FINISHED!!')
        print('\n', now.strftime("%x"), now.strftime("%X"))
        print('=============================================================\n')
        
    else :
        
        print('\n\n=============================================================')
        print('\n   >> Running ' + str(script_name) + '...')
        print('\n', now.strftime("%x"), now.strftime("%X"))
        print('=============================================================\n')
        
    return



# (10) Function to extract labels separated by '_' from a long informative string
def extract_tags (long_string_tag, position='all', extension=True) : 
    
    if extension == True : long_string_tag = long_string_tag[0:-4]
   
    tags   = []
    result = ''
    
    j = 0
    for i in range(0, len(long_string_tag)) : 
        
        if long_string_tag[i] == '_' : 
            
            tags.append(long_string_tag[j:i])
            j = i + 1
            
    tags.append(long_string_tag[j:])
    
    result = tags 
    
    if position != 'all' and position < len(tags) : result = tags[position] 
    
    return result 



# (12) Function to convert the LM expert flags to binary [0,1,-1] == [el,sp,ERROR]
def convert_labels_expert (expert_array, double_column=False) : 
        
    length = len(expert_array)
    
    if double_column == True:
        
        result = np.zeros((length, 2), dtype=int)
        
        for i in range(0, length) : 
                        
            if expert_array[i] == 'L' : result[i, 1] = 1
            
            else : result[i, 0] = 1
            
    else : 
        
        result = np.zeros((length,), dtype=int)
        
        for i in range(0, length) : 
                        
            if expert_array[i] == 'L' : result[i] = 1
                           
    return result



# (13) Function to trim an image keeping it centred
def crop_image_RGB (images_array_original, desired_size) : 

    if images_array_original.shape[1] != images_array_original.shape[2] : 

        print('\n >> ERROR utils_AE(13): the images are not square scaled!!\n')
        sys.exit()
        
    elif images_array_original.shape[1] <= desired_size : 

        print('\n >> ERROR utils_AE(13): incorrect parameter values!!\n')
        sys.exit()
            
    sqr_dim = images_array_original.shape[1]
    
    n_crop = sqr_dim - desired_size
    
    chunk = int(n_crop / 2.0)
    
    pixels_to_crop = [i for i in range(0, chunk)] + [i for i in range(chunk + desired_size, sqr_dim)]

    images_array_original = np.delete(images_array_original, pixels_to_crop, axis=1)
    images_array_original = np.delete(images_array_original, pixels_to_crop, axis=2)  

    return images_array_original     



# (14.1) Function to split whole features and labels arrays into training and validation sets
def data_split_AE_test (images_array, labels_pd, test_partition, n_splits=5) : 
    
    if test_partition > n_splits or test_partition < 1 : 

        print('\n >> ERROR utils_AE(14): the test_partition referenced is incorrect!!\n')
        sys.exit()        
        
    if len(images_array) != len(labels_pd) : 
        
        print('\n >> ERROR utils_AE(14): images and labels arrays length do not match!!\n')
        sys.exit()
       
    data_partitions = [len(x) for x in np.array_split(images_array, n_splits)]
    start_point = sum(data_partitions[0 : (test_partition - 1)])
    end_point   = start_point + data_partitions[test_partition - 1] 
    
    images_train = np.vstack((images_array[0 : start_point], images_array[end_point :]))
    images_test  = images_array[start_point : end_point]
    
    labels_selection = [i for i in range(0, start_point)] + \
    [i for i in range(end_point, len(labels_pd))] 
        
    labels_train = labels_pd.iloc[labels_selection]
    labels_train = labels_train.reset_index()
    labels_train = pd.DataFrame(data=labels_train, columns=list(labels_pd.columns))
   
    labels_test = labels_pd[start_point : end_point]
    labels_test = labels_test.reset_index()
    labels_test = pd.DataFrame(data=labels_test, columns=list(labels_pd.columns))
    
    return (images_train, labels_train, images_test, labels_test)



def vectorize_labels_ERL (labels_array) : 
    
    result = np.zeros((len(labels_array), 3), dtype=int)
    
    for i in range(0, len(labels_array)) : 
        
        if labels_array[i] == 0 : result[i, 0] = 1
        
        elif labels_array[i] == 1 : result[i, 1] = 1 
    
        else : result[i, 2] = 1
        
        
    return result



def antivectorize_labels_ERL (labels_vector) : 
    
    result = np.zeros((len(labels_vector),), dtype=int)
    
    tags = (0, 1, -1)
    
    for i in range(0, len(labels_vector)) : 
        
        max_value = np.argmax(labels_vector[i])
        
        result[i] = tags[max_value]
 
       
    return result



def vectorize_labels_bin (labels_array) : 
    
    result = np.zeros((len(labels_array), 2), dtype=int)
    
    for i in range(0, len(labels_array)) : 
        
        if labels_array[i] == 0 : result[i, 0] = 1
    
        else : result[i, 1] = 1
                
    return result



def antivectorize_labels_bin (labels_vector) : 
    
    result = np.zeros((len(labels_vector),), dtype=int)
    
    tags = (0, 1)
    
    for i in range(0, len(labels_vector)) : 
        
        max_value = np.argmax(labels_vector[i])
        
        result[i] = tags[max_value]
 
       
    return result



# (14.2) Function to split whole features and labels arrays into training and validation sets
def data_split_CNN_test_expert (images_array, labels_pd, test_partition, n_splits=5) : 
    
    if test_partition > n_splits or test_partition < 1 : 

        print('\n >> ERROR utils_AE(14): the test_partition referenced is incorrect!!\n')
        sys.exit()        
        
    if len(images_array) != len(labels_pd) : 
        
        print('\n >> ERROR utils_AE(14): images and labels arrays length do not match!!\n')
        sys.exit()
       
    data_partitions = [len(x) for x in np.array_split(images_array, n_splits)]
    start_point = sum(data_partitions[0 : (test_partition - 1)])
    end_point   = start_point + data_partitions[test_partition - 1] 
    
    images_train = np.vstack((images_array[0 : start_point], images_array[end_point :]))
    images_test  = images_array[start_point : end_point]
    
    labels_selection = [i for i in range(0, start_point)] + \
    [i for i in range(end_point, len(labels_pd))] 
        
    labels_train = labels_pd.iloc[labels_selection]
    labels_train = labels_train.reset_index()
    labels_train = pd.DataFrame(data=labels_train, columns=list(labels_pd.columns))
   
    labels_test = labels_pd[start_point : end_point]
    labels_test = labels_test.reset_index()
    labels_test = pd.DataFrame(data=labels_test, columns=list(labels_pd.columns))
    
    labels_train_vector = np.array(labels_train['EXPERT'], dtype=str)
    labels_train_vector = convert_labels_expert(labels_train_vector, double_column=True)
    
    labels_test_vector = np.array(labels_test['EXPERT'], dtype=str)
    labels_test_vector = convert_labels_expert(labels_test_vector, double_column=True) 
    
    return (images_train, labels_train, labels_train_vector,
            images_test, labels_test, labels_test_vector)

    

# (14.3) Function to split whole features and labels arrays into training and validation sets
def data_split_CNN_test_amateur (images_array, labels_pd, test_partition, n_splits=5) : 
    
    if test_partition > n_splits or test_partition < 1 : 

        print('\n >> ERROR utils_AE(14): the test_partition referenced is incorrect!!\n')
        sys.exit()        
        
    if len(images_array) != len(labels_pd) : 
        
        print('\n >> ERROR utils_AE(14): images and labels arrays length do not match!!\n')
        sys.exit()
       
    data_partitions = [len(x) for x in np.array_split(images_array, n_splits)]
    start_point = sum(data_partitions[0 : (test_partition - 1)])
    end_point   = start_point + data_partitions[test_partition - 1] 
    
    images_train = np.vstack((images_array[0 : start_point], images_array[end_point :]))
    images_test  = images_array[start_point : end_point]
    
    labels_selection = [i for i in range(0, start_point)] + \
    [i for i in range(end_point, len(labels_pd))] 
        
    labels_train = labels_pd.iloc[labels_selection]
    labels_train = labels_train.reset_index()
    labels_train = pd.DataFrame(data=labels_train, columns=list(labels_pd.columns))
   
    labels_test = labels_pd[start_point : end_point]
    labels_test = labels_test.reset_index()
    labels_test = pd.DataFrame(data=labels_test, columns=list(labels_pd.columns))
    
    labels_train_binary = expand_to_double_column(np.array(labels_train['AMATEUR_MAJOR'], dtype=int))
    labels_test_binary  = expand_to_double_column(np.array(labels_test['AMATEUR_MAJOR'], dtype=int))
    
    return (images_train, labels_train, labels_train_binary,
            images_test, labels_test, labels_test_binary)



# (15.1) Function to split training features and labels arrays into training/validation sets
def data_split_AE_training (images_array_train, train_val_ratio=0.7) : 
    
    train_partition = int(train_val_ratio * len(images_array_train))
    
    images_train      = images_array_train[0 : train_partition]
    images_validation = images_array_train[train_partition :]
    
    return (images_train, images_validation)



def data_split_CNN_test (images_array, labels_pd, labels_tag, test_partition, n_splits=5, n_class=2) : 
    
    if test_partition > n_splits or test_partition < 1 : 

        print('\n >> ERROR utils_AE(14): the test_partition referenced is incorrect!!\n')
        sys.exit()        
        
    if len(images_array) != len(labels_pd) : 
        
        print('\n >> ERROR utils_AE(14): images and labels arrays length do not match!!\n')
        sys.exit()
       
    data_partitions = [len(x) for x in np.array_split(images_array, n_splits)]
    start_point = sum(data_partitions[0 : (test_partition - 1)])
    end_point   = start_point + data_partitions[test_partition - 1] 
    
    images_train = np.vstack((images_array[0 : start_point], images_array[end_point :]))
    images_test  = images_array[start_point : end_point]
    
    labels_selection = [i for i in range(0, start_point)] + \
    [i for i in range(end_point, len(labels_pd))] 
        
    labels_train = labels_pd.iloc[labels_selection]
    labels_train = labels_train.reset_index()
    labels_train = pd.DataFrame(data=labels_train, columns=list(labels_pd.columns))
   
    labels_test = labels_pd[start_point : end_point]
    labels_test = labels_test.reset_index()
    labels_test = pd.DataFrame(data=labels_test, columns=list(labels_pd.columns))
    
    if n_class == 2 : 
        
        labels_train_vector = vectorize_labels_bin(np.array(labels_train[labels_tag], dtype=int))
        labels_test_vector  = vectorize_labels_bin(np.array(labels_test[labels_tag], dtype=int))
        
    elif n_class == 3 : 
        
        labels_train_vector = vectorize_labels_ERL(np.array(labels_train[labels_tag], dtype=int))
        labels_test_vector  = vectorize_labels_ERL(np.array(labels_test[labels_tag], dtype=int))
    
    return (images_train, labels_train, labels_train_vector,
            images_test, labels_test, labels_test_vector)



# (15.2) Function to split training features and labels arrays into training/validation sets
def data_split_CNN_training (images_array_train, labels_array_train, train_val_ratio=0.7) : 
    
    if len(images_array_train) != len(labels_array_train) : 
    
        print('\n >> ERROR utils_CNN(15.2): images and labels arrays length do not match!!\n')
        sys.exit()
    
    train_partition = int(train_val_ratio * len(images_array_train))
    
    images_train = images_array_train[0 : train_partition]
    images_val   = images_array_train[train_partition :]
    
    labels_train = labels_array_train[0 : train_partition] 
    labels_val   = labels_array_train[train_partition :] 
    
    return (images_train, labels_train, images_val, labels_val)



def Random_Under_Over_Sampling (images_array_train, images_labels_train, oversampling=False) : 
    
    positive_bool = np.array(antivectorize_labels_bin(images_labels_train), dtype=bool)
    negative_bool = np.invert(positive_bool)
    
    n_positive = sum(positive_bool)
    n_negative = sum(negative_bool)
        
    positive_images = images_array_train[positive_bool]
    positive_labels = images_labels_train[positive_bool]
    
    negative_images = images_array_train[negative_bool]
    negative_labels = images_labels_train[negative_bool]
    
        
    if oversampling == False : 
               
        negative_undersampling = np.random.choice(n_negative, n_positive, replace=False)
        
        negative_images = negative_images[negative_undersampling]
        negative_labels = negative_labels[negative_undersampling]
               
    else : 
        
        positive_oversampling = np.random.choice(n_positive, n_negative, replace=True)
        
        positive_images = positive_images[positive_oversampling]
        positive_labels = positive_labels[positive_oversampling]
        
               
    images_array_modified  = np.vstack((negative_images, positive_images))
    images_labels_modified = np.vstack((negative_labels, positive_labels))
    
    n_examples = len(images_array_modified)
    
    shuffling = np.random.choice(n_examples, n_examples, replace=False)
    
    images_array_modified  = images_array_modified[shuffling]
    images_labels_modified = images_labels_modified[shuffling]
                
    return (images_array_modified, images_labels_modified)


def imbalance_ratio (labels_vector_array) : 
    
    n_positive = sum(antivectorize_labels_bin(labels_vector_array))
    
    ratio = (n_positive * 1.0) / len(labels_vector_array)
    
    return ratio



# (16) Function to convert seconds into dd:hh:mm:ss
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



# (3.2) Function to print to a .txt and .csv files all runnning information and classifier performance
def classification_performance_CNN (array_train, predictions_array, labels_test, exec_time,
                                    output_path, output_name, partition, n_class=2) : 
    
    def global_statistics_performance (output_file_dir, output_filename) : 
        
        csv_name = '_' + output_filename + '_classification.csv'
    
        csv_path = os.path.join(output_file_dir, csv_name)
        csv_file = pd.read_csv(csv_path)
        
        acc  = csv_file['Accuracy']
        time = csv_file['Train_time(s)']
        
        meta_metrics_tags = ['Acc_mean', 'Acc_std', 'Train_time(s)_mean', 'Train_time(s)_std']
        meta_metrics_data = [acc.mean(), acc.std(), time.mean(), time.std()] 
            
        csv_result = pd.DataFrame(data=[meta_metrics_data], columns=meta_metrics_tags)
        
        csv_result_path = os.path.join(output_file_dir, '__' + output_filename + '_summary.csv')
        csv_result.to_csv(csv_result_path, index=False, float_format='%.4f')
    
        return

    if n_class == 2 : 
        
        predictions = antivectorize_labels_bin(predictions_array)
        labels      = antivectorize_labels_bin(labels_test)
        
    elif n_class == 3 : 
        
        predictions = antivectorize_labels_ERL(predictions_array)
        labels      = antivectorize_labels_ERL(labels_test)
        
    
    accuracy = (sum(predictions == labels) * 1.0) / len(predictions)
        
    exec_time = round(exec_time, 2)
    exec_time_str = convert_seconds(exec_time)
    
    list_dir = os.listdir(output_path)  
    csv_metrics_name = '_' + output_name + '_classification.csv'
    csv_metrics_path = os.path.join(output_path, csv_metrics_name)
    
    metrics_tags = ['Test_part', 'N_train', 'N_test', 'Train_time(s)', 'Train_time', 'Accuracy']
    new_raw_data = [partition, len(array_train), len(labels_test), exec_time, exec_time_str,
                    accuracy]

    new_row = pd.DataFrame(data=[new_raw_data], columns=metrics_tags)
   
    if csv_metrics_name not in list_dir : 
        
        csv_metrics = new_row
        
    else : 
        
        csv_metrics = pd.read_csv(csv_metrics_path)    
        csv_metrics = csv_metrics.append(new_row, ignore_index=True)

    
    csv_metrics.to_csv(csv_metrics_path, index=False, float_format='%.4f')
            
    if len(csv_metrics) == 5 : global_statistics_performance(output_path, output_name)

    return 



# (3.4) Function to compute the mean and std of the csv file originated in (3)
def global_statistics_executions (CWD, trial_name, n_executions) : 
    
    summary_global = pd.DataFrame(data=[], columns=['Acc_mean', 'Acc_std', 'Train_time(s)_mean',
                                                    'Train_time(s)_std'])
   
    output_root_dir = os.path.join(CWD, trial_name)
       
    for i in range(1, n_executions + 1) : 
        
        folder_name = trial_name + '_' + str(i)
        folder_path = os.path.join(output_root_dir, folder_name)
        
        summary_csv = '__' + trial_name + '_summary.csv'
        summary_csv = os.path.join(folder_path, summary_csv)
        
        summary_csv = pd.read_csv(summary_csv)
        
        summary_global = summary_global.append(summary_csv)
        
        
    global_statistics_row = [summary_global['Acc_mean'].mean(), summary_global['Acc_mean'].std(),
                         summary_global['Train_time(s)_mean'].mean(), summary_global['Train_time(s)_mean'].std()]
    
    global_statistics = pd.DataFrame(data=[global_statistics_row], columns=['Accs_mean',
                                                                           'Accs_std',
                                                                           'Train_time(s)s_mean',
                                                                           'Train_time(s)s_std'])
    
    summary_global_path    = os.path.join(output_root_dir, '_' + trial_name + '_executions.csv')
    global_statistics_path = os.path.join(output_root_dir, '__' + trial_name + '_executions_summary.csv')
    
    summary_global.to_csv(summary_global_path, index=False, float_format='%.4f')
    global_statistics.to_csv(global_statistics_path, index=False, float_format='%.4f')

    return



def global_statistics_executions_MG (CWD, trial_name, n_executions) : 
    
    summary_global_columns = ['G-mean_mean', 'G-mean_std', 'AUC_mean', 'AUC_std', 'Train_time(s)_mean',
                              'Train_time(s)_std'] 
    
    summary_global = pd.DataFrame(data=[], columns=summary_global_columns)
   
    output_root_dir = os.path.join(CWD, trial_name)
       
    for i in range(1, n_executions + 1) : 
        
        folder_name = trial_name + '_' + str(i)
        folder_path = os.path.join(output_root_dir, folder_name)
        
        summary_csv = '__' + trial_name + '_summary.csv'
        summary_csv = os.path.join(folder_path, summary_csv)
        
        summary_csv = pd.read_csv(summary_csv)
        
        summary_global = summary_global.append(summary_csv)
        
        
    global_statistics_row = [summary_global['G-mean_mean'].mean(), summary_global['G-mean_mean'].std(),
                             summary_global['AUC_mean'].mean(), summary_global['AUC_mean'].std(),
                         summary_global['Train_time(s)_mean'].mean(), summary_global['Train_time(s)_mean'].std()]
    
    global_statistics_columns = ['G-means_mean', 'G-means_std', 'AUCs_mean', 'AUCs_std',
                                 'Train_time(s)s_mean', 'Train_time(s)s_std']
    
    global_statistics = pd.DataFrame(data=[global_statistics_row], columns=global_statistics_columns)
    
    summary_global_path    = os.path.join(output_root_dir, '_' + trial_name + '_executions.csv')
    global_statistics_path = os.path.join(output_root_dir, '__' + trial_name + '_executions_summary.csv')
    
    summary_global.to_csv(summary_global_path, index=False, float_format='%.4f')
    global_statistics.to_csv(global_statistics_path, index=False, float_format='%.4f')

    return



def classification_performance_imbalance (train_array, predictions_array, labels_test, ratio,
                                          exec_time, output_path, output_name, partition) : 
    
    def global_statistics_imbalance (output_file_dir, output_filename) : 
        
        csv_name = '_' + output_filename + '_classification.csv'
    
        csv_path = os.path.join(output_file_dir, csv_name)
        csv_file = pd.read_csv(csv_path)
        
        gmean = csv_file['G-mean']
        auc  = csv_file['AUC']
        time = csv_file['Train_time(s)']
        
        meta_metrics_tags = ['G-mean_mean', 'G-mean_std', 'AUC_mean', 'AUC_std', 'Train_time(s)_mean',
                             'Train_time(s)_std']
        meta_metrics_data = [gmean.mean(), gmean.std(), auc.mean(), auc.std(), time.mean(), time.std()] 
            
        csv_result = pd.DataFrame(data=[meta_metrics_data], columns=meta_metrics_tags)
        
        csv_result_path = os.path.join(output_file_dir, '__' + output_filename + '_summary.csv')
        csv_result.to_csv(csv_result_path, index=False, float_format='%.4f')
        
        return
     
    predictions_array_bin = antivectorize_labels_bin(predictions_array)
    labels_test_bin       = antivectorize_labels_bin(labels_test)
    
    gmean = geometric_mean_score(labels_test_bin, predictions_array_bin) 
    auc   = roc_auc_score(labels_test_bin, predictions_array_bin) 
       
    exec_time = round(exec_time, 2)
    exec_time_str = convert_seconds(exec_time)
    
    list_dir = os.listdir(output_path)  
    csv_metrics_name = '_' + output_name + '_classification.csv'
    csv_metrics_path = os.path.join(output_path, csv_metrics_name)
    
    metrics_tags = ['Test_part', 'N_train', 'N_test', 'Ratio', 'Train_time(s)', 'Train_time', 'G-mean', 'AUC']
    new_raw_data = [partition, len(train_array), len(labels_test), ratio, exec_time, exec_time_str, gmean, auc]

    new_row = pd.DataFrame(data=[new_raw_data], columns=metrics_tags)
   
    if csv_metrics_name not in list_dir : 
        
        csv_metrics = new_row
        
    else : 
        
        csv_metrics = pd.read_csv(csv_metrics_path)    
        csv_metrics = csv_metrics.append(new_row, ignore_index=True)
        
    csv_metrics.to_csv(csv_metrics_path, index=False, float_format='%.4f')
            
    if len(csv_metrics) == 5 : global_statistics_imbalance(output_path, output_name)

    return



# (3.3) Function to print to a .txt and .csv files all runnning information and classifier performance
def save_predictions (data_test, predictions_array, partition, classifier_tag, output_path, output_name) : 
    
    csv_name = output_name + '_test_results_' + str(partition) + '.csv'
    csv_path = os.path.join(output_path, csv_name)
    
    classifier_tag = classifier_tag 
    
    data_test[classifier_tag] = np.array(predictions_array, dtype=int)
    
    data_test = data_test[['OBJID', 'EL_RAW', 'CS_RAW', 'AMATEUR', 'EXPERT', classifier_tag]]
        
    data_test.to_csv(csv_path, index=False)


    return



# (3.4) Function to print to a .txt and .csv files all runnning information and classifier performance
def save_predictions_ES (data_test, predictions_array, partition, output_path, output_name) : 
    
    csv_name = output_name + '_predictions_test_' + str(partition) + '.csv'
    csv_path = os.path.join(output_path, csv_name)
    
    data_test['CNN_EL'] = np.array(predictions_array[:, 0])
    data_test['CNN_CS'] = np.array(predictions_array[:, 1])
       
    data_test.to_csv(csv_path, index=False, float_format='%.3f')

    return



def save_predictions_MG (data_test, predictions_array, partition, output_path, output_name) : 
    
    csv_name = output_name + '_predictions_test_' + str(partition) + '.csv'
    csv_path = os.path.join(output_path, csv_name)
    
    data_test['CNN_NON-MG'] = np.array(predictions_array[:, 0])
    data_test['CNN_MG']     = np.array(predictions_array[:, 1])
       
    data_test.to_csv(csv_path, index=False, float_format='%.3f')

    return



def save_predictions_ERL (data_test, predictions_array, partition, output_path, output_name) : 
    
    csv_name = output_name + '_predictions_test_' + str(partition) + '.csv'
    csv_path = os.path.join(output_path, csv_name)
    
    data_test['CNN_E'] = np.array(predictions_array[:, 0])
    data_test['CNN_R'] = np.array(predictions_array[:, 1])
    data_test['CNN_L'] = np.array(predictions_array[:, 2])
       
    data_test.to_csv(csv_path, index=False, float_format='%.3f')

    return

    

def global_statistics_simpler (output_file_dir, output_filename) : 
        
    csv_name = '_' + output_filename + '_classification.csv'

    csv_path = os.path.join(output_file_dir, csv_name)
    csv_file = pd.read_csv(csv_path)
    
    acc  = csv_file['Accuracy']
    time = csv_file['Train_time(s)']
    
    meta_metrics_tags = ['Acc_mean', 'Acc_std', 'Train_time(s)', 'Train_time(s)_std']
    meta_metrics_data = [acc.mean(), acc.std(), time.mean(), time.std()] 
        
    csv_result = pd.DataFrame(data=[meta_metrics_data], columns=meta_metrics_tags)
    
    csv_result_path = os.path.join(output_file_dir, '__' + output_filename + '_summary.csv')
    csv_result.to_csv(csv_result_path, index=False, float_format='%.4f')
    
    return



# (13) Function to convert a two-column binary vector to a single column binary vector
def round_to_single_column (two_columns_array) : 
    
    length = len(two_columns_array)
    result = np.zeros((length,), dtype=int)
    
    for i in range(0, length) : 
        
        if two_columns_array[i, 1] > two_columns_array[i, 0]   : result[i] = 1
        
    return result



# (2.1) Function to evaluate predictions' Confusion Matrix 
def Confusion_Matrix (predictions_array, labels_array) : 
    
    if len(predictions_array) != len(labels_array) : 
        
        print('\n >> ERROR utils_experiments(2.1): Label arrays length do not match!\n')
        sys.exit()  
    
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    for i in range(0, len(predictions_array)) : 
        
        if predictions_array[i] == 1 and labels_array[i] == 1 : TP += 1
        if predictions_array[i] == 1 and labels_array[i] == 0 : FP += 1
        if predictions_array[i] == 0 and labels_array[i] == 0 : TN += 1
        if predictions_array[i] == 0 and labels_array[i] == 1 : FN += 1
        
    
    return (TP, FN, FP, TN)



# (2.2) Function to evaluate predictions' Accuracy from Confusion Matrix
def Acc (confusion_matrix) : 
    
    trues = confusion_matrix[0] + confusion_matrix[3]
    total = trues + confusion_matrix[1] + confusion_matrix[2]
    
    result = (trues * 1.0) / total
    
    return round(result, 4)



# (2.3) Function to evaluate predictions' Precison from Confusion Matrix
def Precision (confusion_matrix) : 
    
    TP = confusion_matrix[0] * 1.0
    FP = confusion_matrix[2] * 1.0
    
    all_Positive = TP + FP
    
    if all_Positive != 0.0 : precision = TP / all_Positive
    
    else : precision = 0.0
    
    return round(precision, 4)



# (2.4) Function to evaluate predictions' Recall from Confusion Matrix
def Recall (confusion_matrix) : 
    
    TP = confusion_matrix[0] * 1.0
    FN = confusion_matrix[1] * 1.0
    
    denom = TP + FN
    
    if denom != 0.0 : recall = TP / denom
    
    else : recall = 0.0
    
    return round(recall, 4)



# (2.5) Function to evaluate predictions' F1 Score from Confusion Matrix 
def F1_score (confusion_matrix) : 
    
    precision = Precision(confusion_matrix)
    recall    = Recall(confusion_matrix)
    
    denom = precision + recall
    
    if denom != 0.0 : F1_score  = (2 * precision * recall) / denom
    
    else : F1_score = 0.0
    
    return round(F1_score, 4)



# (13.2) Function to convert a two-column binary vector to a single column binary vector
def expand_to_double_column (single_column_array) : 
    
    length = len(single_column_array)
    result = np.zeros((length, 2), dtype=int)
    
    for i in range(0, length) : 
        
        if single_column_array[i] == 0 : result[i, 0] = 1
        
        else : result[i, 1] = 1
        
    return result



def save_script (CWD, script_name, output_folder) : 
    
    script = os.path.basename(script_name)
    
    script_path_destination = os.path.join(CWD, output_folder, script)
    
    shutil.copy(script_name, script_path_destination)
    
    return

     
            

            