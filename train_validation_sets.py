# -*- coding: utf-8 -*-
"""
Train and validation image dataset/folder creation.
Images will be randomly shuffled and copied to the new directory

@author: Osi
"""

import os
import math
import shutil
import random
import numpy as np
import pandas as pd
import tensorflow as tf



speciesCode_list ='A:/Documents/Python Scripts/BirdBot3.0/species_info_111.csv'
speciesCodelist = pd.read_csv(speciesCode_list)
speciesCodelist = speciesCodelist['speciesCode']

# Create new folders to move preprocessed images
#def TrainTestFolder():
for x in ('train', 'validation'):
    folder_name = x
    path = 'A:\Documents\Python Scripts\BirdBot3.0\Training\dataset'
    os.chdir(path)
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
        
    for y in speciesCodelist:
        folder_name = y
        path = f'A:\Documents\Python Scripts\BirdBot3.0\Training\dataset\{x}'
        os.chdir(path)
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

#def EmptyFolderCheck():
empty_folder = []
for z in speciesCodelist:
    folder = f'A:\Documents\Python Scripts\BirdBot3.0\Preprocessing\dataset\{z}'#current working dir of preprocessed images
    file_count = sum(len(files) for _, _, files in os.walk(folder)) #no. of files in dir
    print(f'There are no images in the {z} preprocessing folder')
    empty_folder.append(z)
# If no images in the preprocessing folder, we can either skip that speciesCode in the next loop or pop out of the list


# Randomly move preprocessed images to train, validation, and Test folders
#def TrainTestSplit():
for n in speciesCodelist:
    folder = f'A:\Documents\Python Scripts\BirdBot3.0\Preprocessing\dataset\{n}'#current working dir of preprocessed images
    file_count = sum(len(files) for _, _, files in os.walk(folder)) #no. of files in dir
    dest = f'A:\Documents\Python Scripts\BirdBot3.0\Training\dataset\{n}'
    
    try:
        
        if file_count % 2 > 0:
            train_count = round(file_count*.9)
            validation_count = round(file_count*.1)
            
        
        elif file_count % 2 == 0:
            train_count = round(file_count*.9)
            validation_count = round(file_count*.1)
            
        elif file_count == 0:
            print('The folder corresponding to {n} is empty')
            continue
    
        else:
            pass
    
    except IndexError:
        print('The folder corresponding to {n} is empty') #IndexError: Cannot choose from an empty sequence
        continue
    
    except Exception as e:
        print(e)
        #Error: Destination path 'A:\Documents\Python Scripts\BirdBot3.0\Training\dataset\train\{n}\{n}_#_pad.jpg' already exists
        continue
    
    if file_count != train_count + validation_count:
        train_count = round(file_count*.9) + 1
        validation_count = round(file_count*.1)    

    
    
    #random shuffle and move to new dir        
    for i in range(1, train_count + 1):
        try:
            #Variable random_file stores the name of the random file chosen
            m = 'train'
            random_file = random.choice(os.listdir(folder))
            source_file = f'{folder}\{random_file}'
            dest = f'A:\Documents\Python Scripts\BirdBot3.0\Training\dataset\{m}\{n}'
            dest_file = dest
            #"shutil.move" function moves file from one directory to another
            shutil.move(source_file, dest_file)
        except IndexError:
            print(f'The folder corresponding to {n} is empty') #IndexError: Cannot choose from an empty sequence
            break
    
    for j in range(1, validation_count + 1):
        try:           
            #Variable random_file stores the name of the random file chosen
            m = 'validation'
            random_file = random.choice(os.listdir(folder))
            source_file = f'{folder}\{random_file}'
            dest = f'A:\Documents\Python Scripts\BirdBot3.0\Training\dataset\{m}\{n}'
            dest_file = dest
            #"shutil.move" function moves file from one directory to another
            shutil.move(source_file, dest_file)
        except IndexError:
            print(f'The folder corresponding to {n} is empty') #IndexError: Cannot choose from an empty sequence
            break



# Set the path to the dataset directory
dataset_dir = 'A:/Documents/Python Scripts/BirdBot3.0/Preprocessing/dataset'

# Create the training dataset
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory=dataset_dir + '/train',
    labels='inferred', # Automatically infer the class labels from the directory structure
    label_mode='int', # Integer labels
    image_size=(480, 480), # Resize images to 480x480
    batch_size=32, # Number of images per batch
    shuffle=True, # Shuffle the dataset
    seed=123 # Seed for reproducibility
)

# Create the validation dataset
validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory=dataset_dir + '/validation',
    labels='inferred',
    label_mode='int',
    image_size=(480, 480),
    batch_size=32,
    shuffle=True,
    seed=123
)

# Print the class names
class_names = train_dataset.class_names
print("Class names:", class_names)
