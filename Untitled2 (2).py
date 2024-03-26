#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import CLIPModel
import torch.cuda.amp as amp  # For mixed precision training
from torch.utils.checkpoint import checkpoint_sequential  # For gradient checkpointing


# In[2]:


import torch
import clip


# In[3]:


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing
class EmotionDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data.iloc[idx]
        image = Image.open(image_path).convert('RGB')  # Convert grayscale images to RGB
        if self.transform:
            image = self.transform(image)
        return image, label


# In[4]:


# Define the data directory and emotions
data_dir = "C:\\Users\\osyed\\OneDrive\\Desktop\\Final Year Project\\Combined dataset"
emotions = ["contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise", "anger"]

# Create a DataFrame with image paths and labels
data = pd.DataFrame([(os.path.join(data_dir, emotion, img), emotions.index(emotion))
                     for emotion in emotions
                     for img in os.listdir(os.path.join(data_dir, emotion))
                     if img.endswith(".png") or img.endswith(".jpg")],
                    columns=["image_path", "label"])


# In[5]:


# Split data into train, validation, and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data["label"])
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42, stratify=train_data["label"])


# In[6]:


# Define the transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# In[7]:


# Apply transformations to images
train_dataset = EmotionDataset(train_data, transform=transform)
val_dataset = EmotionDataset(val_data, transform=transform)
test_dataset = EmotionDataset(test_data, transform=transform)


# In[8]:


# DataLoaders
batch_size = 128  # Increase batch size for better GPU utilization
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


# In[9]:


# Load the CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")


# In[10]:


# Move the model to the desired device
model = model.to(device)


# In[11]:


# Freeze CLIP model parameters
for param in model.parameters():
    param.requires_grad = False


# In[12]:


# Define the EmotionDetector model
class EmotionDetector(nn.Module):
    def __init__(self, clip_model, num_classes):
        super(EmotionDetector, self).__init__()
        self.clip_model = clip_model.to(device)  # Move CLIP model to device
        
        # Get the output dimensions by passing a dummy input through the get_image_features method
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            clip_output = self.clip_model.get_image_features(dummy_input)
        self.hidden_size = clip_output.shape[-1]
        
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def forward(self, images):
        clip_output = self.clip_model.get_image_features(images)
        logits = self.classifier(checkpoint_sequential(clip_output, self.classifier))  # Use gradient checkpointing
        return logits

num_classes = len(emotions)
model = EmotionDetector(model, num_classes).to(device)

# Use mixed precision training
scaler = amp.GradScaler()

# Optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)


# In[14]:


# Define the loss criterion
criterion = nn.CrossEntropyLoss()

# Training function
def train_epoch(model, train_loader, optimizer, scaler, device, epoch):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with amp.autocast():  # Mixed precision training
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_loader.dataset)
    print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}")
    return train_loss

# Define the validation function
def validate(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

    val_loss /= len(val_loader.dataset)
    return val_loss


# In[15]:


# Training function
def train_epoch(model, train_loader, optimizer, scaler, device):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        with amp.autocast():  # Mixed precision training
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item() * images.size(0)

    return train_loss / len(train_loader.dataset)


# In[16]:


pip install torchbearer


# In[17]:


from torchbearer.callbacks import EarlyStopping


# In[18]:


# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)


# In[ ]:


# Train the model
num_epochs = 1
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
    val_loss = validate(model, val_loader, device)
    scheduler.step(val_loss)
    if train_epoch(model, optimizer, scheduler, train_loader, val_loader, epoch, early_stopping):
        break


# In[ ]:


# Evaluate the model on the test set
model.eval()
test_acc = 0.0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        test_acc += torch.sum(preds == labels.data)

test_acc /= len(test_loader.dataset)
print(f"Test Accuracy: {test_acc:.4f}")



# In[ ]:


# Save the model
torch.save(model.state_dict(), "emotion_detector2.pth")



# In[ ]:





# In[ ]:




