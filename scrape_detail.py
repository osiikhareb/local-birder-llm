# -*- coding: utf-8 -*-
"""
eBird taxonimic data webscraper using Selenium

@author: Osi
"""

import os
import pandas as pd
import numpy as np
import itertools
from timeit import default_timer as timer
from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.expected_conditions import presence_of_element_located

'''
#https://ebird.org/robots.txt
Crawl delay is 2s for bingbot, Yandex is not allowed on this site
All bots must avoid these directories: /linegraph /barchartData /*checklist
'''


# Later, insert check if species_info_df exists in folder, then check if any search terms are in species_info_df


search_terms ='A:/Documents/Python Scripts/BirdBot3.0/search_terms.csv'
search_terms = pd.read_csv(search_terms)

error_prone_search_terms ='A:/Documents/Python Scripts/BirdBot3.0/error_prone_search_terms.csv'
error_prone_search_terms = pd.read_csv(error_prone_search_terms)



#def scrape_detail(search_terms, error_prone_search_terms):
start = timer()

# Create empty list to store scraped details of each species
species_info = []

# Creating a webdriver instance
driver = webdriver.Chrome()

for i in search_terms:
    query = i
    
    # Open Ebird in the browser
    driver.get('https://ebird.org/explore')
    
    for j in range(1, 2):
        # Initialize action chain
        action = ActionChains(driver)
        
        # Finding the search box using the xpath
        box = driver.find_element(By.XPATH,'//*[@id="Suggest-0"]')

        # Click on the box and wait
        action.click(on_element = box)
        action.pause(2)
        #pause1 = [2,2.2,2.5,2.7,3]
        #action.pause(np.random.choice(pause1))
        
        # Type the search query in the search box
        box.send_keys(query)
        action.pause(2)
        #action.pause(np.random.choice(pause1))
        action.click(on_element = box)
        action.pause(2)
        #action.pause(np.random.choice(pause1))

        if query in error_prone_search_terms:
            # Arrow down to suggested name for species and pres enter
            action.key_down(Keys.DOWN).send_keys().key_up(Keys.DOWN).perform()
            action.pause(1)
            #pause2 = [1,1.2,1.5,1.7,2.1]
            #action.pause(np.random.choice(pause2))
            action.key_down(Keys.RETURN).send_keys().key_up(Keys.RETURN).perform()  #arrow down and arrow up to make sure the first result is always chosen
        
        else:
            # Arrow down to suggested name for species and pres enter
            action.key_down(Keys.DOWN).send_keys().key_up(Keys.DOWN).perform()
            action.pause(1)
            #pause2 = [1,1.2,1.5,1.7,2.1]
            #action.pause(np.random.choice(pause2))
            action.key_down(Keys.UP).send_keys().key_up(Keys.UP).perform()
            action.pause(1)
            action.key_down(Keys.RETURN).send_keys().key_up(Keys.RETURN).perform()  #arrow down and arrow up to make sure the first result is always chosen
            #first_result = driver.find_element(By.XPATH, '//*[@id="Suggest-suggestion-0"]')
            #action.double_click(on_element = first_result)
        '''
        None of the methods above are foolproof due to the first result in rare cases being the result we don't want
        The ultimate fix would be to use bs4 scrape to scrape the search box results
        Look for the <div id="Suggest-suggestion-0"> or id="Suggest-suggestion-1"
        Run an if statment to see whether the <span class "Suggestion-text"> matches the search query
        '''
        
        # Scrape details from species page
        wait = WebDriverWait(driver, 10)
        first_result = wait.until(presence_of_element_located((By.CSS_SELECTOR, "#species-page-heading")))
        taxonomy = first_result.get_attribute("textContent")    #order, family, (sciName) genus, species, (comName) common name; we can use this for class heirarchies?
        print(first_result.get_attribute("textContent"))

        second_result = wait.until(presence_of_element_located((By.CSS_SELECTOR, "p.u-stack-sm:nth-child(2)")))
        description = second_result.get_attribute("textContent")
        print(second_result.get_attribute("textContent"))

        url = driver.current_url    #url to ger species code (speciesCode) or taxon id
        print(driver.current_url)
        
        
        # Process scraped data for efficient storage, this step also serves to "slow" each scrape down
        taxonomy = taxonomy.split()
        
        # Accounting for species names that are 1, 2, or 3 words
        if len(taxonomy) == 6:                
            #order, family, (sciName) genus, species, (comName) common name
            order = taxonomy[0]
            family = taxonomy[1]
            genus = taxonomy[4]
            species = taxonomy[5]
            comName =  taxonomy[2] + ' ' + taxonomy[3]
            sciName = taxonomy[4] + ' ' + taxonomy[5].capitalize()
            #taxonid
            speciesCode = url.split('/')
            speciesCode = speciesCode[4]
        
            # append details to list ; this could be a dictionary instead
            species_info.append([query, speciesCode, comName, sciName, order, family, genus, species, description])
            
        elif len(taxonomy) == 5:
            #order, family, (sciName) genus, species, (comName) common name
            order = taxonomy[0]
            family = taxonomy[1]
            genus = taxonomy[3]
            species = taxonomy[4]
            comName =  taxonomy[2]
            sciName = taxonomy[3] + ' ' + taxonomy[4].capitalize()
            #taxonid
            speciesCode = url.split('/')
            speciesCode = speciesCode[4]
        
            # append details to list
            species_info.append([query, speciesCode, comName, sciName, order, family, genus, species, description])

        elif len(taxonomy) == 7:
            #order, family, (sciName) genus, species, (comName) common name
            order = taxonomy[0]
            family = taxonomy[1]
            genus = taxonomy[5]
            species = taxonomy[6]
            comName =  taxonomy[2] + ' ' + taxonomy[3] + ' ' + taxonomy[4]
            sciName = taxonomy[5] + ' ' + taxonomy[6].capitalize()
            #taxonid
            speciesCode = url.split('/')
            speciesCode = speciesCode[4]
        
            # append details to list
            species_info.append([query, speciesCode, comName, sciName, order, family, genus, species, description])
        
        else:
            #order, family, (sciName) genus, species, (comName) common name
            order = ""
            family = ""
            genus = ""
            species = ""
            comName = ""
            sciName = ""
            #taxonid
            speciesCode = url.split('/')
            speciesCode = speciesCode[4]
        
            # append details to list
            species_info.append([query, speciesCode, comName, sciName, order, family, genus, species, description])
        
        # An implicit(explicit) wait may need to be added here to avoid too many requests
        WebDriverWait(driver, 5)

# Close web driver
driver.quit()   # closes all browser windows and ends driver's session/process
# driver.close() closes the browser active window.

dt = timer() - start
minutes = dt / 60
print(f"Scrape detail took {minutes} minutes to complete")

#speciesCode_list = [el[1] for el in species_info]

#if species_info_df.csv exists:
    #create species_info_df2 then append results to existing df
#else do this:
species_info_df = pd.DataFrame(species_info, columns=['query', 'speciesCode', 'comName', 'sciName', 'order', 'family', 'genus', 'species', 'description'])    
species_info_df.to_csv('A:/Documents/Python Scripts/BirdBot3.0/Scraper/files/species_info.csv')

#return
