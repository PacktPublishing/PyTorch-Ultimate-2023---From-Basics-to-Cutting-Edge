#%% 
# import essentials
# setup image show function
# setup iterator
# setup imgage transformer
# setup data loader
# setup image classification model
# setup loss function and optimizer
# train the model and validate it on the go
# test the model
# save the model

# %% import essentials
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import os


transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean = 0.5, std = 0.5)
])

# %%
batch_size = 2
trainset = torchvision.datasets.ImageFolder(root = 'data/train', transform = transform)
testset = torchvision.datasets.ImageFolder(root = 'data/test', transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True)
testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = True)
classes = ['Positive', 'Negative']
# %%
def imshow(img):
    img = img *0.5 + 0.5  # unnormalize
    npimg = img.numpy()
    print(npimg.shape)
    plt.imshow(npimg.transpose((1, 2, 0)))
    plt.show()

imshow(torchvision.utils.make_grid(next(iter(trainloader))[0], nrow = 2))

# %% Neuralnetwork setup
class ImageClassificationNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(1,10,3) # 10 filters of size 3x3
        self.pool = nn.MaxPool2d(2,2) # 2x2 pooling
        self.conv2 = nn.Conv2d(10, 20, 3) # 20 filters of size 3x3
        self.pool = nn.MaxPool2d(2,2) # 2x2 pooling
        self.fc1 = nn.Linear(20 * 6* 6, 128) 
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,32)
        self.fc4 = nn.Linear(32,1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = torch.flatten(x,1)  # flatten the tensor except for the batch dimension
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x
    
# %% init model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageClassificationNet().to(device)
loss_function = nn.BCELoss() # Binary Cross Entropy Loss for binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001


# %% training
from tqdm import tqdm
NUM_EPOCHS = 20
for epoch in range(NUM_EPOCHS):
    for data in tqdm(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device).float()

        optimizer.zero_grad()  # zero the parameter gradients
        outputs = model(inputs)
        loss = loss_function(outputs, labels.reshape(-1, 1))  # reshape labels to match output shape
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch}/{NUM_EPOCHS}, Loss: {loss.item():.4f}')
# %% test
y_test = []
y_test_hat = []
for data in tqdm(testloader):
    inputs, y_test_temp = data
    inputs = inputs.to(device)
    with torch.no_grad():
        y_test_hat_temp = model(inputs).round()  # round to get binary predictions
    y_test.extend(y_test_temp.numpy())  # move labels to CPU and convert to numpy
    y_test_hat.extend(y_test_hat_temp.numpy())  # move predictions to CPU and convert to numpy

# %% 
accuracy = accuracy_score(y_test, y_test_hat)
print(f'Accuracy: {accuracy*100:.2f}')
# %% save model
torch.save(model.state_dict(), 'binary_classification_model_pk.pth')

# %%
