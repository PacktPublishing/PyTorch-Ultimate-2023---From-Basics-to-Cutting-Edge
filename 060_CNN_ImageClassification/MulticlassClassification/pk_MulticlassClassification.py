#%% import requirements
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


#%% let's view 1 or 2 images
img_path = '/Users/priyanshukhandelwal/Desktop/Learning-Building/Learning/Pytorch/PyTorch-Ultimate-2025/060_CNN_ImageClassification/MulticlassClassification/train/affenpinscher/affenpinscher_12.jpg'
img = Image.open(img_path)
plt.imshow(img)
# %% let's define the transforms
transform = transforms.Compose([
    transforms.Resize((600, 600)),  # Resize the image to 60x60 pixels
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.RandomRotation(10),  # Randomly rotate the image by 10 degrees
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the image
])

# %% let now see the transforms
images_transformed = transform(img)
plt.imshow(images_transformed.squeeze())  # Squeeze to remove the channel dimension for display
print(images_transformed.shape)

# %% let's load the dataset
train_dataset = datasets.ImageFolder('./train', transform=transform)
test_dataset = datasets.ImageFolder('./test', transform=transform)

batch_size = 32  # Define the batch size
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
classes =   train_dataset.classes  # Get the class names
print(f"Classes: {classes}")
# %% lets define CNN model
class MultiClassClassificationModel(nn.Module):
    def __init__(self, num_classes = len(classes)):
        super(MultiClassClassificationModel, self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,) 
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 16, 3)
        self.conv3 = nn.Conv2d(16, 8, 3)
        self.fc1 = nn.Linear(42632, 128 )
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        # print("Conv1 output shape:", x.shape)
        x = self.relu(x)
        x = self.pool(x)
        # print("Pool1 output shape:", x.shape)

        x = self.conv2(x)
        # print("Conv2 output shape:", x.shape)
        x = self.relu(x)
        x = self.pool(x)
        # print("Pool2 output shape:", x.shape)

        x = self.conv3(x)
        # print("Conv3 output shape:", x.shape)
        x = self.relu(x)
        x = self.pool(x)
        # print("Pool3 output shape:", x.shape)

        print(x.shape[1] * x.shape[2] * x.shape[3])
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3]) 
         # Flatten the output
        # print("Flattened output shape:", x.shape)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


# %%
model = MultiClassClassificationModel(num_classes=len(classes))
model.to(device='cuda' if torch.cuda.is_available() else 'cpu')
# %% define loss function and optimizer
loss_function = nn.CrossEntropyLoss()
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# %% training loop
from tqdm import tqdm
NUM_EPOCHS = 10
for epoch in tqdm(range(NUM_EPOCHS)):
    for data in (train_data_loader):
        inputs, labels = data
        inputs = inputs.to(device='cuda' if torch.cuda.is_available() else 'cpu')
        labels = labels.to(device='cuda' if torch.cuda.is_available() else 'cpu')

        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        # print(f'Batch Loss: {loss.item():.4f}')
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.4f}')
    print(loss)
    print(loss.item())


# %% check accuracy on test set
y_test = []
y_test_hat = []
for data in test_data_loader:
    inputs, labels = data
    inputs = inputs.to(device='cuda' if torch.cuda.is_available() else 'cpu')
    labels = labels.to(device='cuda' if torch.cuda.is_available() else 'cpu')

    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)  # Get the index of the max log-probability
    y_test.extend(labels.cpu().numpy())
    y_test_hat.extend(predicted.cpu().numpy())

# %% calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_test_hat)
print(f'Accuracy: {accuracy * 100:.2f}%')

