# -*- coding: utf-8 -*-
"""
Choose and Load a Pretrained Model (e.g., ViT, CLIP, ResNet, EfficientNet)
Fine-tune/retrain the best pretrained model on the image dataset if needed.
This example will be implemented with ResNet50 for now.  
ResNet50 is more accurate than ResNet18, but more computationally expensive and will take longer to train.

@author: Osi
"""


import torch.nn as nn
from torchvision import models
import torch.optim as optim


# Load a pretrained model
model = models.resnet50(pretrained=True)
#model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=len(train_dataset.classes))


# Modify the last layer to match the number of classes in the custom dataset
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)




#Train the model on the image dataset

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()  # Zero out gradients
            
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            
            loss.backward()  # Backpropagate
            optimizer.step()  # Update weights
            
            # Track accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}, Accuracy: {100 * correct / total}%')

train_model(model, train_loader, criterion, optimizer, num_epochs=10)
