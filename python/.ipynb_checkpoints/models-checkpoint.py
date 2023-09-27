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
    
    

# class FashionCNN(nn.Module):
#     def __init__(self):
#         super(FashionCNN, self).__init__()

#         # First convolutional layer
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

#         # Second convolutional layer
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
#         self.bn2 = nn.BatchNorm2d(64)

#         # Third convolutional layer
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
#         self.bn3 = nn.BatchNorm2d(128)

#         # Fully connected layers
#         self.fc1 = nn.Linear(128 * 4 * 4, 256)
#         self.fc_bn1 = nn.BatchNorm1d(256)
        
#         self.fc2 = nn.Linear(256, 128)
#         self.fc_bn2 = nn.BatchNorm1d(128)
        
#         self.fc3 = nn.Linear(128, 10)

#         # Dropout to prevent overfitting
#         self.dropout = nn.Dropout(p=0.5)

#     def forward(self, x):
#         # First convolutional layer
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))

#         # Second convolutional layer
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))

#         # Third convolutional layer
#         x = F.relu(self.bn3(self.conv3(x)))  # Removed pooling

#         # Flatten the tensor
#         x = x.view(-1, 128 * 4 * 4)

#         # First fully connected layer
#         x = F.relu(self.fc_bn1(self.fc1(x)))
#         x = self.dropout(x)

#         # Second fully connected layer
#         x = F.relu(self.fc_bn2(self.fc2(x)))
#         x = self.dropout(x)

#         # Third fully connected layer (output)
#         x = self.fc3(x)

#         return F.log_softmax(x, dim=1)


# class FashionCNN(nn.Module):
#     def __init__(self):
#         super(FashionCNN, self).__init__()
        
#         # First convolutional layer
#         self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        
#         # Second convolutional layer
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                
#         # Dense (fully connected) layers
#         self.fc1 = nn.Linear(64 * 7 * 7, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2, stride=2)
        
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, stride=2)
        
#         # Flatten the tensor
#         x = x.view(x.size(0), -1)
        
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
        
#         return F.log_softmax(x, dim=1)

class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                
        # Dense (fully connected) layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, stride=2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, stride=2)
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        
        # Apply dropout after the first dense layer's activation function
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


