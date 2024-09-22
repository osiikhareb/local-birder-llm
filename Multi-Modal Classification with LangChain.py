# -*- coding: utf-8 -*-
"""
Tie it all together

@author: Osi
"""

#pip install langchain torch torchvision ollama
from PIL import Image
import torch
from torchvision import models, transforms
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain



# Define Text-Based LLM Query Handling

# Initialize the Ollama LLM
llm = Ollama(model="llama")  # Use any local LLM model from Ollama

# Define a prompt template for answering text queries
text_prompt = PromptTemplate(
    input_variables=["query"],
    template="You are an expert bird classification assistant. Answer the question: {query}"
)

# Create an LLM chain for text queries
text_chain = LLMChain(
    llm=llm,
    prompt=text_prompt
)

# Text query example
def handle_text_query(query):
    return text_chain.run(query)

# Example query
text_query = "What bird is light brown with stripes on its back and a red patch on its head?"
text_response = handle_text_query(text_query)
print(f"Text response: {text_response}")




# Define Image-Based Query Handling

# Load pretrained ResNet model for image classification
image_model = models.resnet50(pretrained=True)		#change to trained model
image_model.eval()  # Set model to evaluation mode

# Image transformation for ResNet
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Image classification function
def classify_image(image_path):
    image = Image.open(image_path)
    image = image_transform(image).unsqueeze(0)  # Apply transformations and add batch dimension

    with torch.no_grad():
        output = image_model(image)  # Forward pass

    # Get the predicted class
    _, predicted = torch.max(output, 1)
    return f"Predicted class index: {predicted.item()}"

# Example image classification
image_path = "path_to_your_image.jpg"
image_response = classify_image(image_path)
print(f"Image response: {image_response}")




# Combine Text and Image Tasks in a LangChain Pipeline

from langchain.tools import Tool

# Define text handler as a LangChain Tool
text_tool = Tool(
    name="Text Query Handler",
    description="Handles text queries using a local LLM.",
    func=handle_text_query
)

# Define image handler as a LangChain Tool
image_tool = Tool(
    name="Image Classifier",
    description="Classifies images using a pretrained model.",
    func=classify_image
)

# Define a LangChain Pipeline that chooses between text and image tools
class MultiModalChain:
    def __init__(self, text_tool, image_tool):
        self.text_tool = text_tool
        self.image_tool = image_tool

    def run(self, input_data):
        if isinstance(input_data, str):
            # Handle text queries
            return self.text_tool.func(input_data)
        elif isinstance(input_data, Image.Image):
            # Handle image queries
            return self.image_tool.func(input_data)
        else:
            return "Unsupported input type"

# Instantiate the multi-modal pipeline
multi_modal_chain = MultiModalChain(text_tool, image_tool)

# Example text query
text_query = "What bird is light brown with stripes on its back and a red patch on its head?"
text_response = multi_modal_chain.run(text_query)
print(f"Multi-modal text response: {text_response}")

# Example image query
image_path = "A:\Documents\Python Scripts\BirdBot3.0\unseen_img_data\new_img.jpg"
image_response = multi_modal_chain.run(image_path)
print(f"Multi-modal image response: {image_response}")

