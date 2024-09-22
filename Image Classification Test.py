# -*- coding: utf-8 -*-
"""
Image classification test with trained models

@author: Osi
"""

from PIL import Image

def classify_image(image_path, model):
    model.eval()  # Set model to evaluation mode
    
    # Load the image
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0)  # Apply transformations and add batch dimension
    
    # Move to GPU if available
    img = img.to(device)
    
    with torch.no_grad():
        outputs = model(img)  # Forward pass
        _, predicted = torch.max(outputs.data, 1)  # Get predicted class index
    
    class_idx = predicted.item()
    class_name = train_dataset.classes[class_idx]
    
    print(f'Predicted class: {class_name}')

# Classify a sample image
classify_image('A:\Documents\Python Scripts\BirdBot3.0\unseen_img_data\new_img.jpg', model)	#path to new image unseen by the model




# Test using LangChain

from langchain.tools import Tool

def classify_image_tool(image_path):
    return classify_image(image_path, model)

image_classification_tool = Tool(
    name="Image Classifier",
    description="Classifies an image based on the custom-trained model",
    func=classify_image_tool
)

