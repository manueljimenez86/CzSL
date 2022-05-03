# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 17:24:51 2018

@author: psxmaji
"""

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import cv2
import os
import sys
from datetime import datetime

#matplotlib.use('Agg') 
#plt.switch_backend('agg')

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
    
    result_path = os.path.join(output_folder, trial_name + '.csv')
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
def read_images_tensor (IDs_array, folder_path, dim_tuple, file_ext='.png') :
    
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
        
        image_name = IDs_array[i] + file_ext
        image_path = os.path.join(folder_path, folder, image_name)
        
        image = cv2.imread(image_path, RGB)
        image = image.reshape(dim_tuple)
        
        images_array[i] = image.astype('float32') / 255.0
                
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
        
        if i % 10000 == 0 and i != 0 : print(' -> ' + str(i) + ' images done...')
        
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



#(2.3) Function to read flat RGB images
def read_images_flat_by_ID (IDs_array, IDs_array_to_read, folder_path, dim_tuple, file_ext='.tiff') :
    
    """
    This function reads the sets of 1,000 (RGB) images present in sub-folders of 
    'parent folder', following the order in 'IDs_array'. 'dim_tuple' is the 
    tensorial dimensionality of the image in the format (width, height, channels).
    """
            
    images_array = np.zeros((len(IDs_array_to_read), dim_tuple[0] * dim_tuple[1]), dtype=float)           
    
    folders = sorted([x for x in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, x))])
    
    print('\nLoading images...\n')
    
    for i in range(0, len(IDs_array_to_read)) : 
        
        if i % 10000 == 0 and i != 0 : print(' -> ' + str(i) + ' images done...')
        
        image_index = np.where(IDs_array == IDs_array_to_read[i])
        image_index = image_index[0][0]
        
        image_folder = folders[int(image_index / 1000)]
                    
        image_name = IDs_array_to_read[i] + file_ext
        image_path = os.path.join(folder_path, image_folder, image_name)
        
        image = cv2.imread(image_path, 0).flatten()
        images_array[i] = image.astype('float32') / 255.0
                
    print('\n\nImages load successful!\n')
          
    return images_array



# (3) Function to plot a given sample of original and reconstructed images
def sample_plot (images_original, images_decoded, images_selection, IDs_array,
                 dim_tuple, folder_name, trial_name, title) : 
    
    """
    This function plots an 'image_selection' of IDs, comparing the original image taken from 
    'image_arrary' with the reconstruction performed by the AE in 'image_array_decoded'. The 
    plot is entitled with the 'title' string.
    """
    n = len(images_selection)
    # Seek of images selection location in the original and decoded image arrays
    index_array = np.zeros(n, dtype='int32')
    
    for i in range(0, n) : 
      
        loc = np.where(IDs_array == images_selection[i][:-5])
        index_array[i] = loc[0][0]
              
    # Plot of the images, original and reconstructed        
    plt.figure(figsize=(20, 4))
    plt.suptitle(title, fontsize=20, y=1.0)
    
    i = 0
    if dim_tuple[-1] == 3 : 
    
        for index in index_array:
            
            # Display original image
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(cv2.cvtColor(images_original[index].reshape(dim_tuple), cv2.COLOR_BGR2RGB))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        
            # Display reconstruction
            ax2 = plt.subplot(2, n, i + 1 + n)
            plt.imshow(cv2.cvtColor(images_decoded[index].reshape(dim_tuple), cv2.COLOR_BGR2RGB))
            plt.gray()
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)
            
            i += 1
            
    else : 
        
        for index in index_array:
            
            # Display original image
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(images_original[index].reshape(dim_tuple[0], dim_tuple[1]))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        
            # Display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(images_decoded[index].reshape(dim_tuple[0], dim_tuple[1]))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
            i += 1
        
    path_figure = os.path.join(folder_name, trial_name + '_sample.png')     
     
    plt.savefig(path_figure)
    
    return



# (4) Function to create a directory for output files (OK)
def trial_directory(trial_name) :
    
    """
    This function creates a new directory named 'trial_name' to bring together the output files 
    of the experiment running. If the directory already exists, it creates a new one adding '(x)'
    to the name, with 'x' being an increasing natural number.
    """
    
    done = False
    length = len(trial_name)
    n = 3
    
    while done == False : 
              
        if os.path.isdir(trial_name) == True and len(trial_name) == length : 
            
            trial_name += '(2)'
            
        elif os.path.isdir(trial_name) == True :  
            
            trial_name = trial_name[:-3] + '(' + str(n) + ')'
            n += 1
            
        else : 
            
            os.makedirs(trial_name)
            done = True
                        
    return trial_name



# (5) Function to concatenate words for file names
def concat_str (str_list, file_ext='', separator='_') : 
    
    result = str_list[0]
    
    for i in range(1, len(str_list)) : 
        
        if type(str_list[i]) != str : str_list[i] = str(str_list[i])
        
        if str_list[i] != '' : result = result + separator + str_list[i]
        
    if file_ext != '' : result = result + file_ext
    
    return result



# (6) Function to check the number of arguments passed to the script 
def args_check (args_number) : 
    
    if len(sys.argv) != args_number + 1 : 
    
        print('\n >> ERROR utils_AE(6): incorrect number of arguments passed to the script file!\n')
        sys.exit()
        
    return



# (7) Function to create a directory for output files with several controls
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
                
                directory_name = directory_name[:-3] + '(' + str(n) + ')'
                n += 1
                
            else : 
                
                directory_path = os.path.join(root_path, directory_name)
                os.makedirs(directory_path)
                done = True
                        
    return directory_path



# (8) Function to add a counter at the end of a filename
def add_counter (name_original, counter) : 
    
    if name_original[-4] == '.' : 
        
        name = name_original[:-4]
        extension = name_original[-4:]
        
        new_name = name + '_' + str(counter) + extension
        
    else : 
        
        new_name = name_original + '_' + str(counter)
    
    return new_name



# (9.1) Function to print informative headers to the screen while running the scripts
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



# (9.2) Function to print the current time
def print_time () : 
 
    now = datetime.now()        
    print('\n ', now.strftime("%x"), now.strftime("%X"), '\n')

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



# (11) Function to select files from a listdir variable in function of a filename header
def select_files (directory, header) : 
    
    filelist = os.listdir(directory)
    
    result = []
    
    for i in range(0, len(filelist)) : 
        
        if extract_tags(filelist[i], 0) == header : result.append(filelist[i])
        
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



# (14.1) Function to split whole features and labels arrays into training/test sets
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



# (14.2) Function to split the IDs array into training/test sets
def labels_split_test (labels_pd, test_partition, n_splits=5) : 
    
    data_partitions = [len(x) for x in np.array_split(labels_pd, n_splits)]
    start_point = sum(data_partitions[0 : (test_partition - 1)])
    end_point   = start_point + data_partitions[test_partition - 1] 
    
    labels_selection = [i for i in range(0, start_point)] + \
    [i for i in range(end_point, len(labels_pd))] 
    
    labels_train = labels_pd.iloc[labels_selection]
    labels_train = labels_train.reset_index()
    labels_train = pd.DataFrame(data=labels_train, columns=list(labels_pd.columns))
   
    labels_test = labels_pd[start_point : end_point]
    labels_test = labels_test.reset_index()
    labels_test = pd.DataFrame(data=labels_test, columns=list(labels_pd.columns))
    
    return (labels_train, labels_test)
    
    
    
# (15) Function to split training features and labels arrays into training/validation sets
def data_split_AE_training (images_array_train, train_val_ratio=0.7) : 
    
    train_partition = int(train_val_ratio * len(images_array_train))
    #np.random.shuffle(images_array_train)
    
    images_train      = images_array_train[0 : train_partition]
    images_validation = images_array_train[train_partition :]
    
    return (images_train, images_validation)


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

          
            