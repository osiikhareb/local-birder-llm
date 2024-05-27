# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 19:07:07 2024

@author: Osi
"""

"""
Preprocessing needs:
Read images to a new folder and label them
Get all images to be the same size either by resizing downwards or zero padding
"""

import os
import shutil
import random
import math
import numpy as np
import pandas as pd
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt


#path = 'A:/Documents/Python Scripts/BirdBot2.0/Scraper/images'
#os.chdir(path)

## Load all images as dataset

# Get species code list for looping
speciesCode_list ='A:/Documents/Python Scripts/BirdBot2.0/Scraper/_output/species_info_111.csv'
speciesCodelist = pd.read_csv(speciesCode_list)
speciesCodelist = speciesCodelist['speciesCode']

# Create new folders for preprocessed images
for x in speciesCodelist:
    folder_name = x
    path = 'A:\Documents\Python Scripts\BirdBot2.0\Preprocessing\_images'
    os.chdir(path)
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)

# Check distribution of image sizes and label if possible
imgdata = []
for i in speciesCodelist:    
    for j in range(1, 330):
        try:
            folder = f'A:/Documents/Python Scripts/BirdBot2.0/Scraper/_images/{i}'
            os.chdir(folder)
            
            img = cv.imread(f'{i}_{j}.jpg') 
          
            # fetching the dimensions 
            wid = img.shape[1]
            hgt = img.shape[0]
            pixels = wid * hgt
            MP = pixels/1000000
            AR = wid / hgt
            
            # displaying the dimensions 
            print(str(wid) + "x" + str(hgt))
            
            #birddata = ''
            
            imgdata.append([i, wid, hgt, MP, AR])
        except AttributeError:  #last image in species folder reached
           break 

imgdata_df = pd.DataFrame(imgdata, columns=['speciesCode', 'width', 'height', 'megapixels', 'aspect ratio'])    
imgdata_df.to_csv('A:/Documents/Python Scripts/BirdBot2.0/Preprocessing/_output/species_info.csv')
    
# width, height, resolution, and aspect ratio distributions
plt.hist(imgdata_df['width'])
imgdata_df['width'].value_counts()[:20].plot(kind='barh')

plt.hist(imgdata_df['height'])
imgdata_df['height'].value_counts()[:20].plot(kind='barh')

plt.hist(imgdata_df['megapixels'])
#imgdata_df['megapixels'].value_counts()[:20].plot(kind='barh')

plt.hist(imgdata_df['aspect ratio'])
#imgdata_df['aspect ratio'].value_counts()[:20].plot(kind='barh')

# heat map : species on one axis, resolution on another axis, counts 



# images are downloaded as compressed jpgs so data is already lost compared to png

# Define function to pad image to starndard 480 x 480 square
def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

#path = 'A:/Documents/Python Scripts/BirdBot2.0/Preprocessing/_images'
#os.chdir(path)

# Loop through all images and resize to average dimensions or zero pad, store in new folder
for i in speciesCodelist:
    
    folder = f'A:/Documents/Python Scripts/BirdBot2.0/Scraper/_images/{i}'
    file_count = sum(len(files) for _, _, files in os.walk(folder)) + 1
    #file_count = sum(1 for _, _, files in os.walk(folder) for f in files) + 1
    
    dest = f'A:/Documents/Python Scripts/BirdBot2.0/Preprocessing/_images/{i}'
    dest_file_count = sum(len(files) for _, _, files in os.walk(dest))
    
    if dest_file_count == 0:
        pass
    
    elif dest_file_count > 0:
        continue
    
    for j in range(1, file_count):
        try:
            
            # Zero pad image to 480 x 480 square
            img = Image.open(f'A:/Documents/Python Scripts/BirdBot2.0/Scraper/_images/{i}/{i}_{j}.jpg')            
            size = img.size
            maxpad = (480,480)
            
            #use image resolution to determine how much padding is necessary
            padsize = np.subtract(maxpad, size)
            padwid = padsize[0].item() / 2
            padhgt = padsize[1].item() / 2
            
            if padwid % 2 > 0:
                right = math.floor(padwid)
                left = math.floor(padwid) + 1
                top = int(padhgt)
                bottom = int(padhgt)

            elif padhgt % 2 > 0:
                right = int(padwid)
                left = int(padwid)
                top = math.floor(padhgt)
                bottom = math.floor(padhgt) + 1            

            else:
                right = int(padwid)
                left = int(padwid)
                top = int(padhgt)
                bottom = int(padhgt)                
            
            #pad
            img_new = add_margin(img, top, right, bottom, left, (128, 0, 64))
            img_new = img_new.convert('RGB')
            img_new.save(f'A:/Documents/Python Scripts/BirdBot2.0/Preprocessing/_images/{i}/{i}_{j}_pad.jpg', quality=100)
            #img_new.show()            
            
        except FileNotFoundError:  #if file not found go to next iteration
           continue

