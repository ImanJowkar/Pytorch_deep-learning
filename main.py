import numpy as np
import torch
import cv2
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


# ===============================================device property=============================================

print(torch.__version__)
# print(torch.cuda.get_device_name())
# print(torch.cuda.get_device_properties())
# print(torch.cuda.get_device_capability())




# =============================================== Data Loader============================================











# ================================================Initialization===========================================

batch_size = 256
num_class = 10


train_dataset = torchvision.datasets.MNIST('MNIST', train  = True, transform = transforms.ToTensor(),
 target_transform = None, download = True)


test_dataset = torchvision.datasets.MNIST('MNIST', train  = False, transform = transforms.ToTensor(), 
 target_transform = None, download = True)


train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)



# for batch_number, (images, labels) in enumerate(train_loader):
#     print(f'Batch Number:{batch_number}, \t Image_shape: {images.shape}')


one_train_batch_imgs, one_train_batch_lbls = next(iter(train_loader))
print(one_train_batch_imgs.shape)
print(one_train_batch_lbls.shape)

print(one_train_batch_imgs[0].shape)
image = one_train_batch_imgs[0]
image = image.cpu().detach().numpy()
print(type(image))
print(image.shape)
image = image.reshape(28, 28, 1)
print(image.shape)

cv2.imshow('one_sample', image)
cv2.waitKey(0)


# ==============================================Model==========================================
# nn.Sequential(nn.Conv2d(1, 32, 3), 
#               nn.ReLU(),
#               nn.MaxPool2d(3, 2), 
#               nn.Conv2d(32, 64, 3), 
#               nn.ReLU(),
#               nn.MaxPool2d(3, 2), 
#               nn.Linear(64*7*7, 1024), 
#               nn.Linear(1024, 10))




class CNN(nn.Module):

    def __init__(self, num_class):
        super(CNN, self).__init__()

        # Layer 1
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(3, 2)
        
        # Layer 2
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(3, 2)

        # Layer 3
        self.fc1 = nn.Linear(64*7*7, 1024)
        self.fc2 = nn.Linear(1024, num_class)

device =  'cuda' torch.cuda.is_available else 'cpu'
print(device)
        
 


    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.maxpool1(y)

        y = self.conv2(y)
        y = self.relu2(y)
        y = self.maxpool2(y)

        #flat
        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        y = self.fc2(y)
        
        return y


model = CNN(num_class)
print(model)

# print(model.conv2)
# print(model.conv2)

# print(model.fc2)

# print(model.conv2.weight[0])
# print(model.conv2.bias[0])

device =  'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# # Single GPU or CPU
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.to(device)
# # If it is multi GPU
# if torch.cuda.device_count() > 1:
#   model = nn.DataParallel(model，device_ids=[0,1,2])
# model.to(device)