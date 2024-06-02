# -*- coding: utf-8 -*-
"""
Contrastive Language Image Pretraining (CLIP)
Implemented using OpenAI pretrained model 

@author: Osi
"""

import os
import torch
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torchvision import models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import CLIPProcessor, CLIPModel


# Create the image dataset using scraped/processed images 

# Create a cutsom dataset class
class CustomCLIPDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        text_path = os.path.join(self.image_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        with open(text_path, 'r') as f:
            text = f.read().strip()
        
        return image, text
    
    
# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),    # Resize images to 224x224 to avoid model issues for now - We can keep this at 480x480 later
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create the training and validation datasets
train_dataset = CustomCLIPDataset(image_dir='A:/Documents/Python Scripts/BirdBot3.0/Preprocessing/dataset/train', transform=transform)
validation_dataset = CustomCLIPDataset(image_dir='A:/Documents/Python Scripts/BirdBot3.0/Preprocessing/dataset/validation', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)





# Training and fine-tuning

# Load the processor and the pre-trained model
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.train()


# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Define the optimizer for fine-tuning on the custom dataset
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, texts in train_loader:
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)

        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text

        labels = torch.arange(logits_per_image.size(0), device=device)
        loss_img = torch.nn.functional.cross_entropy(logits_per_image, labels)
        loss_txt = torch.nn.functional.cross_entropy(logits_per_text, labels)
        loss = (loss_img + loss_txt) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, texts in validation_loader:
            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)

            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text

            preds = logits_per_image.argmax(dim=1)
            labels = torch.arange(logits_per_image.size(0), device=device)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")

print("Training complete")
