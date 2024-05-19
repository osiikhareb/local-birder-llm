# Local Birder LLM

## Overview

Using data for birds local to Arizona, image classification and NLP will be performed using locally run LLMs.  Using scraped species information and images from eBird.com, LLMs run on the host machine will be used to classify both images and text queries.  Due to the narrower scope and amount of data, SLMs will also be explored.

Using the Selenium web driver, ~1,000 images for roughly 111 different birds native to Arizona will be scraped from the Cornell Lab Macaulay library yielding a training dataset of over 100,000 images.  The scraped images have varying resolutions and will be padded to a standard 480x480 square.  A vision transformer (ViT) will be used to classify each of the 111 classes.  While convolutional neural networks (CNN) are the usual choice in image classification, this serves as an exploration of the broad use cases of transformer models.  Vision transformers are known to be both data-hungry and computationally expensive.  Fine-tuning will be needed to increase accuracy, but this approach may not yield the most accurate classification.  Again, this is just a proof of concept for a small locally run case.

Along with images, taxonomic information and species descriptions for each class will be scraped for all 111 classes.  This information will serve as the basis for a small dataset that can be utilized by a retrieval augmented generation (RAG) architecture. A prompt will be entered, e.g. "What bird is light brown with stripes on its back and a red patch on its head".  The results from the RAG using the stored data should produce the result "Gila Woodpecker".  If the species information data is insufficient, additional information will be scraped from Wikipedia or the detailed description page on eBird.

The overall goal is an exploration of locally obtained data run on local LLMs/SLMs for a specific use case without the need for APIs.  Datasets such as the Caltech-UCSD Birds-200-2011 (CUB-200-2011) exist, but have fewer images per class and aren't specific to Arizona though there is some overlap!  High accuracy has been achieved on the smaller CalTech dataset (11,788 images) using few-shot learning. Given that there aren't millions of training images, that approach may be explored later with a truncated dataset.      
