import torch
import torch.nn as nn
import torch.nn.functional as F


# One hidden Layer NN
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(784, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        x = x.view((-1, 784))
        h = F.relu(self.fc(x))
        h = self.fc2(h)
        return F.log_softmax(h,dim=-1)  
    
    


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Convolution 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        
        # Convolution 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        
        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(64 * 12 * 12, 100)
        
        # Fully connected 2 (readout)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        
        # Convolution 1
        x = self.conv1(x)
        x = F.relu(x)
        
        # Convolution 2 
        x = self.conv2(x)
        x = F.relu(x)
        
        # Max pooling (subsampling)
        x = F.max_pool2d(x, kernel_size=2)
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Fully connected 1
        x = self.fc1(x)
        x = F.relu(x)
        
        # Fully connected 2
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
    
    

class SimplestCNN(nn.Module):
    def __init__(self):
        super(SimplestCNN, self).__init__()
        
        # Single Convolution layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)
        
        # Single Fully connected layer (readout)
        self.fc1 = nn.Linear(16 * 12 * 12, 10)  # after max-pooling, the 28x28 image becomes 12x12

    def forward(self, x):
        
        # Single Convolution followed by ReLU and Max Pooling
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Single Fully Connected Layer
        x = self.fc1(x)
        
        return F.log_softmax(x, dim=1)
