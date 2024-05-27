# -*- coding: utf-8 -*-
"""
Image scraper using Selenium Web driver.
Over 1000 images are scraped from each of the 100+ classes taken from the eBird species code list.

@author: Osi
"""

import pandas as pd
import numpy as np
import os
from timeit import default_timer as timer
import bs4
import requests
from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.expected_conditions import presence_of_element_located


'''
#https://www.macaulaylibrary.org/robots.txt
No bots are allowed to touch this directory /wp-admin/ everything else is fair game
'''


    
# list of species codes, point to stored location and take slice of column
speciesCode_list ='A:/Documents/Python Scripts/BirdBot3.0/Scraper/files/species_info_111.csv'
speciesCodelist = pd.read_csv(speciesCode_list)
speciesCodelist = speciesCodelist['speciesCode']


def ImgScrapeDir():    
    # Creating a directory to save images
    for x in speciesCodelist:
        folder_name = x
        path = 'A:\Documents\Python Scripts\BirdBot3.0\Scraper\_images'
        os.chdir(path)
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)
    
def download_image(url, folder_name, num):
    # write image to file
    reponse = requests.get(url)
    if reponse.status_code==200:
        with open(os.path.join(folder_name, str(folder_name)+"_"+str(num)+".jpg"), 'wb') as file:
            file.write(reponse.content)

#def ScrapeImage(speciesCodelist):    
    start = timer()
    
    # Creating a webdriver instance
    #driver = webdriver.Chrome()
    driver = webdriver.Firefox()
    driver.maximize_window()
    
    # Macaulay link where images will be scraped; we can potentially add tags for labeled photos 
    for i in speciesCodelist:
        
        # Set working folder to have images downloaded to
        folder_name = i
        
        # Loop through species codes on Macaulay Library sorted by best quality images
        search_url = f"https://media.ebird.org/catalog?taxonCode={i}&sort=rating_rank_desc&mediaType=photo"
        driver.get(search_url)
        
        # Initialize action chain
        action = ActionChains(driver)
        
        # Click the "More results" button to show more photos
        # Each click corresponds to 5 images per row where there a 6 rows loaded per page click (30 images per click)
        # We start with 30 images loaded
        n = 0
        while n <= 35:   #change this number to set number of clicks
            errors = [TimeoutException]
            wait = WebDriverWait(driver, timeout=60, poll_frequency=5, ignored_exceptions=errors)
            button = wait.until(presence_of_element_located((By.CSS_SELECTOR, ".pagination > button:nth-child(1)")))
            button.click()
            #action.pause(5)
            #pause1 = [10,10.4,10.7,11,11.3,11.6,12]
            #action.pause(np.random.choice(pause1))
            n += 1
            '''
        n = 0
        while n <= 9:   #change this number to set number of clicks
            try:
                wait = WebDriverWait(driver, 35)
                button = wait.until(presence_of_element_located((By.CSS_SELECTOR, ".pagination > button:nth-child(1)")))
                button.click()
                WebDriverWait(driver, 15)
                #action.pause(5)
                #pause1 = [10,10.4,10.7,11,11.3,11.6,12]
                #action.pause(np.random.choice(pause1))
                n += 1
            except TimeoutException:   #if webdriver stops, then break out of all loops and quit        
                driver.refresh()
                wait = WebDriverWait(driver, 35)
                button = wait.until(presence_of_element_located((By.CSS_SELECTOR, ".pagination > button:nth-child(1)")))
                button.click()
                #action.pause(5)
                pause1 = [10,10.4,10.7,11,11.3,11.6,12]
                action.pause(np.random.choice(pause1))
                n += 1
                continue
       '''
        # Calculate the number of gallery rows for looping
        len_gallery_rows = (n+1)*6
        total_images = len_gallery_rows * 5
        counter = 1 #counter is the image number that we are currently working on scraping
        
        i = 1
        j = 1
        
        for i in range(1, len_gallery_rows+1):
            for j in range(1, 6):
                try:
                 # Select image (i,j)
                 imageElement = driver.find_element(By.CSS_SELECTOR, f'div.ResultsGallery-row:nth-child({i}) > a:nth-child({j}) > figure:nth-child(1) > img:nth-child(1)')
                 # Get the source image link
                 imageURL = imageElement.get_attribute("src")
                except NoSuchElementException:  #if there are less than 5 elements in a row, continue
                     continue
                
                # Set one of these depending on the desired image resolution
                #imageURL = imageURL.replace('/480', '/480 ') #pad first 480 to avoid potentially trunctuating image id when altering
                #imageURL = imageURL.replace('/480 ', '')   #slightly higher res (640x427)
                #imageURL = imageURL.replace('/480 ', '/1200')  #higher res (1200x800)
                #imageURL = imageURL.replace('/480 ', '/2400')  #even higher res! (2400x1600)
                
                # Download images
                try:
                    download_image(imageURL, folder_name, counter)
                    print(f"Success! Downloaded image ({i},{j}) i.e. image {counter} out of {total_images} total. URL: {imageURL}")
                except:
                    print(f"Error downloading image ({i},{j}) i.e. image {counter}, proceeding to the next image")
                
                counter += 1
    
    # Close web driver
    driver.quit()   # closes all browser windows and ends driver's session/process
    # driver.close() closes the browser active window.
                
    dt = timer() - start
    minutes = dt / 60
    print(f"scrape-image.py took {minutes} minutes to complete")
		
    #return




#if __name__ == "__main__":
#    main()