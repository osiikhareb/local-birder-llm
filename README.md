# Local Birder LLM/SLM

## Overview

Using data on birds local to Arizona, image classification and NLP will be performed using locally run LLMs.  Using scraped species information and images from eBird.com, LLMs run on the host machine will be used to classify both images and text queries.  Due to the narrower scope and amount of data, SLMs will also be explored.  Again, the narrow use case of this project and the goal of being run locally makes this a good test for Retrieval Augmented Generation (RAG).

Using the Selenium web driver, ~1,000 images for roughly 111 different birds native to Arizona will be scraped from the Cornell Lab Macaulay library yielding a training dataset of over 100,000 images.  The scraped images have varying resolutions and will be padded to a standard 480x480 square.  A vision transformer (ViT) will be used to classify each of the 111 classes.  While convolutional neural networks (CNN) are the usual choice in image classification, this serves as an exploration of the broad use cases of transformer models.  Vision transformers are known to be both data-hungry and computationally expensive.  Fine-tuning will be needed to increase accuracy, but this approach may not yield the most accurate classification.  Again, this is just a proof of concept for a small locally run case.  The model approach is subject to change depending on the accuracy of the LLM chosen.  While the ViT will be explored first, other models such as CLIP (Contrastive Language-Image Pre-training), ResNet (Residual Networks), as well as EfficientNet and MobileNet will also be used.

Along with images, taxonomic information and species descriptions for each class will be scraped for all 111 classes.  This information will serve as the basis for a small dataset (knowledge base) that can be utilized by a RAG architecture.  Additional knowledge libraries will be added using scraped species descriptions from Wikipedia and/or the detailed description pages on eBird.  The operation should work as follows: A prompt will be entered, e.g. "What bird is light brown with stripes on its back and a red patch on its head".  The results from the RAG LLM using the stored data should produce the result "Gila Woodpecker".

The overall goal is an exploration of locally obtained data run on local LLMs/SLMs for a specific use case without the need for APIs.  Datasets such as the Caltech-UCSD Birds-200-2011 (CUB-200-2011) exist, but have fewer images per class and aren't specific to Arizona though there is some overlap!  High accuracy has been achieved on the smaller CalTech dataset (11,788 images) using few-shot learning. Given that there aren't millions of training images, that approach may be explored later with a truncated dataset.      

## Roadmap

- [ ] Use Selenium to scrape species images and species descriptions
- [ ] Build vision transformer in PyTorch
- [ ] Build other Models in PyTorch
- [ ] Train on image dataset and compare the performance of each
- [ ] Select the best model and Fine-tune
- [ ] Build RAG Architecture
  - [ ] Create a small vector database from species descriptions
  - [ ] Add additional knowledge libraries
    - [ ] Use Selenium to scrape Wikipedia and eBird (again)
  - [ ] Deploy local LLM (Ollama most likely)
  - [ ] Train model on database
  - [ ] Retrain on SLM
- [ ] Test with queries and fine-tune model
- [ ] LangChain to make it all work together?
- [ ] TBD
