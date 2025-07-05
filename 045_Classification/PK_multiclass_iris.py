#%%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

#%% import data
data = load_iris()
X = data.data
y = data.target

#%% train test split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, )


# %%
X_train.size
# %%
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('int64')
y_test = y_test.astype('int64')
# %%
class IrisDataset(Dataset):
    def __init__(self, X, y) -> None:
        super().__init__()
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.len
    
# %%
train_data_set = IrisDataset(X = X_train, y = y_train)
test_data_set = IrisDataset(X = X_test, y = y_test)

train_data_loader = DataLoader(dataset=train_data_set, batch_size=32, shuffle=True)
test_data_loader = DataLoader(dataset=test_data_set, batch_size=32, shuffle=False)
# %% Check dims
print(f"X Shape: {train_data_set.X.shape}, y shape: {train_data_set.y.shape}")

# %% Create MultiClassNet class
class MultiClassNet(nn.Module):
    def __init__(self, NUM_FEATURES, NUM_CLASSES, HIDDEN_FEATURES):
        super().__init__()
        self.lin1 = nn.Linear(NUM_FEATURES, HIDDEN_FEATURES)
        # self.lin2 = nn.Linear(HIDDEN_FEATURES, HIDDEN_FEATURES//2)
        self.lin2 = nn.Linear(HIDDEN_FEATURES, NUM_CLASSES)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self,x):
        x = self.lin1(x)
        x = torch.sigmoid(x)
        x = self.lin2(x)
        # x = torch.sigmoid(x)
        # x = self.lin3(x)
        x = self.log_softmax(x)
        return x
    
    # def forward(self, x):
    #     x = self.lin1(x)
    #     x = torch.sigmoid(x)
    #     x = self.lin2(x)
    #     x = self.log_softmax(x)
    #     return x

# %% Hyperparameters
NUM_FEATURES = train_data_set.X.shape[1]
NUM_CLASSES = len(np.unique(y_train))
HIDDEN_FEATURES = 6
# %%model
model = MultiClassNet(NUM_FEATURES=NUM_FEATURES, NUM_CLASSES=NUM_CLASSES, HIDDEN_FEATURES=HIDDEN_FEATURES)
# %%loss function
loss_function = nn.CrossEntropyLoss()
# %% optimizer
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# %% training loop
NUM_EPOCHS = 5000
train_loss = []
val_loss = []
train_accuracy = []
val_accuracy = []

for epoch in range(NUM_EPOCHS):
    model.train()
    for X_batch, y_batch in train_data_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = loss_function(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_data_loader:
            outputs = model(X_batch)
            loss = loss_function(outputs, y_batch)
            val_loss.append(loss.item())
            preds = torch.argmax(outputs, dim=1)
            accuracy = accuracy_score(y_batch, preds)
            train_accuracy.append(accuracy)
            val_accuracy.append(accuracy)
    print(f"Epoch: {epoch}, Train Loss: {np.mean(train_loss)}, Val Loss: {np.mean(val_loss)}, Train Accuracy: {np.mean(train_accuracy)}, Val Accuracy: {np.mean(val_accuracy)}")





# %%
sns.lineplot(x= range(len(train_loss[-200:])), y = train_loss[-200:])

# %%
sns.lineplot(x= range(len(val_loss[-200:])), y = val_loss[-200:])
# %%
sns.lineplot(x= range(len(train_accuracy[-200:])), y = train_accuracy[-200:])
# %% testing model seperately
X_test_torch = torch.from_numpy(X_test)
with torch.no_grad():
    y_test_hat = model(X_test_torch)
    y_test_hat_index = torch.max(y_test_hat.data, 1)
# %% Accuracy
accuracy = accuracy_score(y_test, y_test_hat_index.indices)
print(f"Test Accuracy: {accuracy}")
# %%
torch.save(model.state_dict(), 'multiclass_iris_model_pk.pth')
# %%
