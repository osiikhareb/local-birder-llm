# -*- coding: utf-8 -*-
"""
Residual Neural Network (ResNet)
Pre-trained ResNet18 model used for image classification
We eventually can go up to ResNet50 depending on initial accuracy of ResNet18

@author: Osi
"""

import os
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import Adam
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR



# Create the image dataset using scraped/processed images
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.endswith(('jpg', 'png', 'jpeg')):
                    self.image_files.append(os.path.join(root, file))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        label = os.path.basename(os.path.dirname(img_path))
        label = int(label)  # Assuming labels are integer class indices
        if self.transform:
            image = self.transform(image)
        return image, label





# Create the modeling class using pre-trained ResNet18 model
class ResNetClassifier:
    def __init__(self, num_classes, learning_rate=0.001, step_size=7, gamma=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model = self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)
    
    def train(self, train_loader, val_loader, num_epochs=25):
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
            self.scheduler.step()
            
            val_acc = self.evaluate(val_loader)
            print(f"Validation Accuracy: {val_acc:.2f}%")

    def evaluate(self, val_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total
    
    


# Training and validation

# Define transformations for the training and validation sets
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create the training and validation datasets
train_dataset = CustomImageDataset(image_dir='A:/Documents/Python Scripts/BirdBot3.0/Preprocessing/dataset/train', transform=train_transform)
val_dataset = CustomImageDataset(image_dir='A:/Documents/Python Scripts/BirdBot3.0/Preprocessing/dataset/validation', transform=val_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Create the model
num_classes = len(os.listdir('A:/Documents/Python Scripts/BirdBot3.0/Preprocessing/dataset/train'))
classifier = ResNetClassifier(num_classes=num_classes)

# Train and evaluate the model
classifier.train(train_loader, val_loader, num_epochs=25)

