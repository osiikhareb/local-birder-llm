# -*- coding: utf-8 -*-
"""
Vision Transformer (pre-trained) using ImageNet weights
Loaded from Hugging Face transformers library

@author: Osi
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchvision import models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor


# Training and validation dataset instantiation

# Set the path to the dataset directory
dataset_dir = 'A:/Documents/Python Scripts/BirdBot3.0/Preprocessing/dataset'

# Define transformations for the training and validation sets
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 to avoid model issues for now - We can keep this at 480x480 later
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize to ImageNet standards
])

# Create the training dataset
train_dataset = datasets.ImageFolder(root=dataset_dir + '/train', transform=transform)

# Create the validation dataset
validation_dataset = datasets.ImageFolder(root=dataset_dir + '/validation', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

# Print the class names
class_names = train_dataset.classes
print("Class names:", class_names)

# Optionally augment the data to create additional images to improve model generalization 
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomRotation(10),  # Randomly rotate images by 10 degrees
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])




# Model instansiation and training

# Load the feature extractor and the pre-trained model
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=len(train_dataset.classes))
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))  # Number of classes


# Define the machine
machine = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(machine)

# Define the optimizer and loss function
optimizer = Adam(model.parameters(), lr=1e-4)
criterion = CrossEntropyLoss()

# Training epochs
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = feature_extractor(images=inputs, return_tensors="pt")['pixel_values']
        inputs, labels = inputs.to(machine), labels.to(machine)

        optimizer.zero_grad()

        outputs = model(pixel_values=inputs).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs = feature_extractor(images=inputs, return_tensors="pt")['pixel_values']
            inputs, labels = inputs.to(machine), labels.to(machine)

            outputs = model(pixel_values=inputs).logits
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")

print("Training complete")



# Evaluation of model performance

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in validation_loader:
        inputs = feature_extractor(images=inputs, return_tensors="pt")['pixel_values']
        inputs, labels = inputs.to(machine), labels.to(machine)

        outputs = model(pixel_values=inputs).logits
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Final Validation Accuracy: {accuracy:.2f}%")



