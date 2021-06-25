# This Deep Learning model was created on 25 Jun 2021 for the EVA 6 course and implements two specific theoretical concepts
# The Depthwise separable convolutions ,Coarse Cutouts as specialized augmentation strategies and dilated kernels with feature concatenation. 
# Developers  - Bharath Kumar Bolla, Dinesh, Manu and Sabeesh
import torch
from torchvision import datasets
from google.colab import drive
from tqdm import tqdm_notebook
from PIL import Image
import numpy as np
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A

class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()

# FIRST MAJOR BLOCK
        # first convolutional block
        self.convblock1a = nn.Sequential(
            nn.Conv2d(in_channels = 3,out_channels = 16, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        # second convolutional block  - normal layer
        self.convblock1b = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels = 16, groups = 1, dilation  = 1,padding = 1,kernel_size= (3,3)),         # in 32, out 32, RF 5
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        # Using dilations instead of maxpooling
        self.convpool1 = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 2,padding = 2,kernel_size= (3,3),stride = 2)          # in 32, out 16, RF ?
        )

# SECOND MAJOR BLOCK
          # first convolutional block
        self.convblock2a = nn.Sequential(
            # nn.Conv2d(in_channels = 16,out_channels = 32, dilation  = 1,padding = 1,kernel_size= (3,3)),                    # in 16, out 16, RF ?
            nn.Conv2d(in_channels = 16,out_channels = 16,groups = 16, dilation  = 1,padding = 1,kernel_size= (3,3)),
            nn.Conv2d(in_channels = 16,out_channels = 32, dilation = 1,padding = 0,kernel_size= (1,1)), # 8, 8, 3  
            nn.ReLU(),
            nn.BatchNorm2d(32)
        ) # will the learning will be better in depth wise or in normal 3x3 convolutions?

        # second convolutional block - 
        self.convblock2b = nn.Sequential(
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 2,padding = 1,kernel_size= (3,3)),                    # in 16, out 14, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        # Using dilation to simulate max pooling
        self.convpool2 = nn.Sequential(
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 2,padding = 2,kernel_size= (3,3),stride = 2)          # in 14, out 7, RF ?
        )

# THIRD MAJOR BLOCK
         # first convolutional block
        self.convblock3a = nn.Sequential(
            # nn.Conv2d(in_channels = 32,out_channels = 64, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 32,out_channels = 32,groups = 32, dilation  = 1,padding = 1,kernel_size= (3,3)),        # in 7, out 7, RF ?
            nn.Conv2d(in_channels = 32,out_channels = 64, dilation = 1,padding = 0,kernel_size= (1,1)), # 8, 8, 3           # in 7, out 7, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        # second convolutional block
        self.convblock3b = nn.Sequential(
            # nn.Conv2d(in_channels = 48,out_channels = 64, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 64,out_channels = 64,groups = 64, dilation  = 1,padding = 1,kernel_size= (3,3)),        # in 7, out 7, RF ?
            nn.Conv2d(in_channels = 64,out_channels = 128, dilation  = 1,padding = 0,kernel_size= (1,1)),                    # in 7, out 7, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        # using dilation to simulate maxpooling
        self.convpool3 = nn.Sequential(
            nn.Conv2d(in_channels = 128,out_channels = 128, dilation  = 2,padding = 2,kernel_size= (3,3),stride = 2)          # in 7, out 4, RF ?
        )

# FOURTH MAJOR BLOCK
         # first convolutional block
        # self.convblock4a = nn.Sequential(
        #     nn.Conv2d(in_channels = 128,out_channels = 128, dilation  = 1,padding = 1,kernel_size= (3,3)),                   # in 4, out 4, RF ?
        #     nn.ReLU(),
        #     nn.BatchNorm2d(128)
        # )
        # second convolutional block
        self.convblock4b = nn.Sequential(
            # nn.Conv2d(in_channels = 128,out_channels = 128, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 128,out_channels = 128, groups = 128, dilation  = 1,padding = 1,kernel_size= (3,3)),    # in 4, out 4, RF ?
            nn.Conv2d(in_channels = 128,out_channels = 256, dilation  = 1,padding = 0,kernel_size= (1,1)),                  # in 4, out 4, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
# GAP LAYER
        self.gap = nn.AvgPool2d(4)                                                                                          # in 4, out 1, RF ?
        self.convblockf = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 


    def forward(self, x):
        x = self.convblock1a(x)
        x = self.convblock1b(x)
        x = self.convpool1(x)
        # print(x.shape)

        x = self.convblock2a(x)
        x = self.convblock2b(x)
        x = self.convpool2(x)
        # print(x.shape)        
        
        x = self.convblock3a(x)
        x = self.convblock3b(x)
        x = self.convpool3(x)

        x = self.convblock4b(x)
        x = self.gap(x)
        # print(x.shape)

        x = self.convblockf(x)
        # print(x.shape)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class CifarNet2(nn.Module):
    def __init__(self):
        super(CifarNet2, self).__init__()

# FIRST MAJOR BLOCK
        # first convolutional block
        self.convblock1a = nn.Sequential(
            nn.Conv2d(in_channels = 3,out_channels = 16, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        self.convblock1b = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        # Using dilations instead of maxpooling
        self.convpool1 = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 2,padding = 2,kernel_size= (3,3),stride = 2)          # in 32, out 16, RF ?
        )

# SECOND MAJOR BLOCK
          # first convolutional block
        self.convblock2a = nn.Sequential(
            # nn.Conv2d(in_channels = 16,out_channels = 32, dilation  = 1,padding = 1,kernel_size= (3,3)),                    # in 16, out 16, RF ?
            nn.Conv2d(in_channels = 16,out_channels = 16,groups = 16, dilation  = 1,padding = 1,kernel_size= (3,3)),
            nn.Conv2d(in_channels = 16,out_channels = 32, dilation = 1,padding = 0,kernel_size= (1,1)), # 8, 8, 3  
            nn.ReLU(),
            nn.BatchNorm2d(32)
        ) # will the learning will be better in depth wise or in normal 3x3 convolutions?

        self.convblock2b = nn.Sequential(
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

        # Using dilation to simulate max pooling
        self.convpool2 = nn.Sequential(
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 2,padding = 2,kernel_size= (3,3),stride = 2)          # in 16, out 8, RF ?
        )

# THIRD MAJOR BLOCK
         # first convolutional block
        self.convblock3a = nn.Sequential(
            # nn.Conv2d(in_channels = 32,out_channels = 64, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 32,out_channels = 32,groups = 32, dilation  = 1,padding = 1,kernel_size= (3,3)),        # in 8, out 8, RF ?
            nn.Conv2d(in_channels = 32,out_channels = 64, dilation = 1,padding = 0,kernel_size= (1,1)), # 8, 8, 3           # in 8, out 8, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.convblock3b = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 2,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        
        # using dilation to simulate maxpooling
        self.convpool3 = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 2,padding = 2,kernel_size= (3,3),stride = 2)          # in 8, out 4, RF ?
        )

# FOURTH MAJOR BLOCK
        # second convolutional block
        self.convblock4a = nn.Sequential(
            # nn.Conv2d(in_channels = 128,out_channels = 128, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 64,out_channels = 64, groups = 64, dilation  = 1,padding = 1,kernel_size= (3,3)),    # in 4, out 4, RF ?
            nn.Conv2d(in_channels = 64,out_channels = 128, dilation  = 1,padding = 0,kernel_size= (1,1)),                  # in 4, out 4, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.convblock4b = nn.Sequential(
            # nn.Conv2d(in_channels = 128,out_channels = 128, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 128,out_channels = 128, groups = 128, dilation  = 1,padding = 1,kernel_size= (3,3)),    # in 4, out 4, RF ?
            nn.Conv2d(in_channels = 128,out_channels = 256, dilation  = 1,padding = 0,kernel_size= (1,1)),                  # in 4, out 4, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )


# GAP LAYER
        self.gap = nn.AvgPool2d(3)                                                                                          # in 4, out 1, RF ?
        self.convblockf = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 


    def forward(self, x):
        x = self.convblock1a(x)
        x = self.convblock1b(x)
        x = self.convpool1(x)
        x = self.convblock2a(x)
        x = self.convblock2b(x)
        x = self.convpool2(x)
        x = self.convblock3a(x)
        x = self.convblock3b(x)
        x = self.convpool3(x)
        x = self.convblock4a(x)
        x = self.convblock4b(x)
        x = self.gap(x)
        x = self.convblockf(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


class CifarNet3(nn.Module):
    def __init__(self):
        super(CifarNet3, self).__init__()

# FIRST MAJOR BLOCK
        # first convolutional block
        self.convblock1a = nn.Sequential(
            nn.Conv2d(in_channels = 3,out_channels = 16, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        self.convblock1b = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        self.convblock1c = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        # Using dilations instead of maxpooling
        self.convpool1 = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 2,padding = 2,kernel_size= (3,3),stride = 2)          # in 32, out 16, RF ?
        )

# SECOND MAJOR BLOCK
          # first convolutional block
        self.convblock2a = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels = 16,groups = 16, dilation  = 1,padding = 1,kernel_size= (3,3)),
            nn.Conv2d(in_channels = 16,out_channels = 32, dilation = 1,padding = 0,kernel_size= (1,1)),                     # in 16, out 16, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(32)
        ) # will the learning will be better in depth wise or in normal 3x3 convolutions?

        self.convblock2b = nn.Sequential(
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.convblock2c = nn.Sequential(
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

        # Using dilation to simulate max pooling
        self.convpool2 = nn.Sequential(
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 2,padding = 2,kernel_size= (3,3),stride = 2)          # in 16, out 8, RF ?
        )

# THIRD MAJOR BLOCK
         # first convolutional block
        self.convblock3a = nn.Sequential(
            # nn.Conv2d(in_channels = 32,out_channels = 64, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 32,out_channels = 32,groups = 32, dilation  = 1,padding = 1,kernel_size= (3,3)),        # in 8, out 8, RF ?
            nn.Conv2d(in_channels = 32,out_channels = 64, dilation = 1,padding = 0,kernel_size= (1,1)), # 8, 8, 3           # in 8, out 8, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.convblock3b = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.convblock3c = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        
        # using dilation to simulate maxpooling
        self.convpool3 = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 2,padding = 2,kernel_size= (3,3),stride = 2)          # in 8, out 4, RF ?
        )

# FOURTH MAJOR BLOCK
        # second convolutional block
        self.convblock4a = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64, groups = 64, dilation  = 1,padding = 1,kernel_size= (3,3)),    # in 4, out 4, RF ?
            nn.Conv2d(in_channels = 64,out_channels = 128, dilation  = 1,padding = 0,kernel_size= (1,1)),                  # in 4, out 4, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.convblock4b = nn.Sequential(
            nn.Conv2d(in_channels = 128,out_channels = 128, groups = 128, dilation  = 1,padding = 1,kernel_size= (3,3)),    # in 4, out 4, RF ?
            nn.Conv2d(in_channels = 128,out_channels = 256, dilation  = 1,padding = 0,kernel_size= (1,1)),                  # in 4, out 4, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )

# GAP LAYER
        self.gap = nn.AvgPool2d(4)                                                                                          # in 4, out 1, RF ?
        self.convblockf = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 

    def forward(self, x):
        x = self.convblock1a(x)
        x = self.convblock1b(x)
        x = self.convblock1c(x)
        x = self.convpool1(x)
        x = self.convblock2a(x)
        x = self.convblock2b(x)
        x = self.convblock2c(x)
        x = self.convpool2(x)
        x = self.convblock3a(x)
        x = self.convblock3b(x)
        x = self.convblock3c(x)
        x = self.convpool3(x)
        x = self.convblock4a(x)
        x = self.convblock4b(x)
        x = self.gap(x)
        x = self.convblockf(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class CifarNet4(nn.Module):
    def __init__(self):
        super(CifarNet4, self).__init__()

# FIRST MAJOR BLOCK
        # first convolutional block
        self.convblock1a = nn.Sequential(
            nn.Conv2d(in_channels = 3,out_channels = 16, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        self.convblock1b = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        # Using dilations instead of maxpooling
        self.convpool1 = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 1,padding = 1,kernel_size= (3,3),stride = 2)          # in 32, out 16, RF ?
        )

# SECOND MAJOR BLOCK
          # first convolutional block
        self.convblock2a = nn.Sequential(
            # nn.Conv2d(in_channels = 16,out_channels = 32, dilation  = 1,padding = 1,kernel_size= (3,3)),                    # in 16, out 16, RF ?
            nn.Conv2d(in_channels = 16,out_channels = 16,groups = 16, dilation  = 1,padding = 1,kernel_size= (3,3)),
            nn.Conv2d(in_channels = 16,out_channels = 32, dilation = 1,padding = 0,kernel_size= (1,1)), # 8, 8, 3  
            nn.ReLU(),
            nn.BatchNorm2d(32)
        ) # will the learning will be better in depth wise or in normal 3x3 convolutions?

        self.convblock2b = nn.Sequential(
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

        # Using dilation to simulate max pooling
        self.convpool2 = nn.Sequential(
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 1,padding = 1,kernel_size= (3,3),stride = 2)          # in 16, out 8, RF ?
        )

# THIRD MAJOR BLOCK
         # first convolutional block
        self.convblock3a = nn.Sequential(
            # nn.Conv2d(in_channels = 32,out_channels = 64, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 32,out_channels = 32,groups = 32, dilation  = 1,padding = 1,kernel_size= (3,3)),        # in 8, out 8, RF ?
            nn.Conv2d(in_channels = 32,out_channels = 64, dilation = 1,padding = 0,kernel_size= (1,1)), # 8, 8, 3           # in 8, out 8, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.convblock3b = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        
        # using dilation to simulate maxpooling
        self.convpool3 = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 1,padding = 1,kernel_size= (3,3),stride = 2)          # in 8, out 4, RF ?
        )

# FOURTH MAJOR BLOCK
        # second convolutional block
        self.convblock4a = nn.Sequential(
            # nn.Conv2d(in_channels = 128,out_channels = 128, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 64,out_channels = 64, groups = 64, dilation  = 1,padding = 1,kernel_size= (3,3)),    # in 4, out 4, RF ?
            nn.Conv2d(in_channels = 64,out_channels = 128, dilation  = 1,padding = 0,kernel_size= (1,1)),                  # in 4, out 4, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.convblock4b = nn.Sequential(
            # nn.Conv2d(in_channels = 128,out_channels = 128, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 128,out_channels = 128, groups = 128, dilation  = 1,padding = 1,kernel_size= (3,3)),    # in 4, out 4, RF ?
            nn.Conv2d(in_channels = 128,out_channels = 256, dilation  = 1,padding = 0,kernel_size= (1,1)),                  # in 4, out 4, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )


# GAP LAYER
        self.gap = nn.AvgPool2d(4)                                                                                          # in 4, out 1, RF ?
        self.convblockf = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 


    def forward(self, x):
        x = self.convblock1a(x)
        x = self.convblock1b(x)
        x = self.convpool1(x)
        x = self.convblock2a(x)
        x = self.convblock2b(x)
        x = self.convpool2(x)
        x = self.convblock3a(x)
        x = self.convblock3b(x)
        x = self.convpool3(x)
        x = self.convblock4a(x)
        x = self.convblock4b(x)
        x = self.gap(x)
        x = self.convblockf(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class CifarNet5(nn.Module):
    def __init__(self):
        super(CifarNet5, self).__init__()

# FIRST MAJOR BLOCK
        # first convolutional block
        self.convblock1a = nn.Sequential(
            nn.Conv2d(in_channels = 3,out_channels = 16, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        self.convblock1b = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        # Using stride 2 without dilation instead of maxpooling
        self.convpool1 = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 1,padding = 1,kernel_size= (3,3),stride = 2), # in 32, out 16, RF ?
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 1,padding = 0,kernel_size= (1,1),stride = 1)   # to check if this can compensate for the mx pooling feature loss       
        )


# SECOND MAJOR BLOCK
          # first convolutional block
        self.convblock2a = nn.Sequential(
            # nn.Conv2d(in_channels = 16,out_channels = 32, dilation  = 1,padding = 1,kernel_size= (3,3)),                    # in 16, out 16, RF ?
            nn.Conv2d(in_channels = 16,out_channels = 16,groups = 16, dilation  = 1,padding = 1,kernel_size= (3,3)),
            nn.Conv2d(in_channels = 16,out_channels = 32, dilation = 1,padding = 0,kernel_size= (1,1)), # 8, 8, 3  
            nn.ReLU(),
            nn.BatchNorm2d(32)
        ) # will the learning will be better in depth wise or in normal 3x3 convolutions?

        self.convblock2b = nn.Sequential(
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

        # Using stride 2 without dilation to simulate max pooling
        self.convpool2 = nn.Sequential(
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 1,padding = 1,kernel_size= (3,3),stride = 2),
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 1,padding = 0,kernel_size= (1,1),stride = 1)          # in 16, out 8, RF ?
        )

# THIRD MAJOR BLOCK
         # first convolutional block
        self.convblock3a = nn.Sequential(
            # nn.Conv2d(in_channels = 32,out_channels = 64, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 32,out_channels = 32,groups = 32, dilation  = 1,padding = 1,kernel_size= (3,3)),        # in 8, out 8, RF ?
            nn.Conv2d(in_channels = 32,out_channels = 64, dilation = 1,padding = 0,kernel_size= (1,1)), # 8, 8, 3           # in 8, out 8, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.convblock3b = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        
        # using dilation to simulate maxpooling
        self.convpool3 = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 1,padding = 1,kernel_size= (3,3),stride = 2),          # in 8, out 4, RF ?
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 1,padding = 0,kernel_size= (1,1),stride = 1)
        )

# FOURTH MAJOR BLOCK
        # second convolutional block
        self.convblock4a = nn.Sequential(
            # nn.Conv2d(in_channels = 128,out_channels = 128, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 64,out_channels = 64, groups = 64, dilation  = 1,padding = 1,kernel_size= (3,3)),    # in 4, out 4, RF ?
            nn.Conv2d(in_channels = 64,out_channels = 128, dilation  = 1,padding = 0,kernel_size= (1,1)),                  # in 4, out 4, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.convblock4b = nn.Sequential(
            # nn.Conv2d(in_channels = 128,out_channels = 128, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 128,out_channels = 128, groups = 128, dilation  = 1,padding = 1,kernel_size= (3,3)),    # in 4, out 4, RF ?
            nn.Conv2d(in_channels = 128,out_channels = 256, dilation  = 1,padding = 0,kernel_size= (1,1)),                  # in 4, out 4, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )


# GAP LAYER
        self.gap = nn.AvgPool2d(4)                                                                                          # in 4, out 1, RF ?
        self.convblockf = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 


    def forward(self, x):
        x = self.convblock1a(x)
        x = self.convblock1b(x)
        x = self.convpool1(x)
        x = self.convblock2a(x)
        x = self.convblock2b(x)
        x = self.convpool2(x)
        x = self.convblock3a(x)
        x = self.convblock3b(x)
        x = self.convpool3(x)
        x = self.convblock4a(x)
        x = self.convblock4b(x)
        x = self.gap(x)
        x = self.convblockf(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class CifarNet6(nn.Module):
    def __init__(self):
        super(CifarNet6, self).__init__()

# FIRST MAJOR BLOCK
        # first convolutional block
        self.convblock1a = nn.Sequential(
            nn.Conv2d(in_channels = 3,out_channels = 16, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        self.convblock1b = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        # Using dilations instead of maxpooling
        self.convpool1 = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 2,padding = 1,kernel_size= (10,10),stride = 1)          # in 32, out 16, RF ?
        )

# SECOND MAJOR BLOCK
          # first convolutional block
        self.convblock2a = nn.Sequential(
            # nn.Conv2d(in_channels = 16,out_channels = 32, dilation  = 1,padding = 1,kernel_size= (3,3)),                    # in 16, out 16, RF ?
            nn.Conv2d(in_channels = 16,out_channels = 16,groups = 16, dilation  = 1,padding = 1,kernel_size= (3,3)),
            nn.Conv2d(in_channels = 16,out_channels = 32, dilation = 1,padding = 0,kernel_size= (1,1)), # 8, 8, 3  
            nn.ReLU(),
            nn.BatchNorm2d(32)
        ) # will the learning will be better in depth wise or in normal 3x3 convolutions?

        self.convblock2b = nn.Sequential(
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 16, out 16, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

        # Using dilation to simulate max pooling
        self.convpool2 = nn.Sequential(
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 2,padding = 0,kernel_size= (5,5),stride = 1)          # in 16, out 8, RF ?
        )

# THIRD MAJOR BLOCK
         # first convolutional block
        self.convblock3a = nn.Sequential(
            # nn.Conv2d(in_channels = 32,out_channels = 64, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 32,out_channels = 32,groups = 32, dilation  = 1,padding = 1,kernel_size= (3,3)),        # in 8, out 8, RF ?
            nn.Conv2d(in_channels = 32,out_channels = 64, dilation = 1,padding = 0,kernel_size= (1,1)), # 8, 8, 3           # in 8, out 8, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.convblock3b = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 8, out 8, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        
        # using dilation to simulate maxpooling
        self.convpool3 = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 2,padding = 0,kernel_size= (3,3),stride = 1)          # in 8, out 4, RF ?
        )

# FOURTH MAJOR BLOCK
        # second convolutional block
        self.convblock4a = nn.Sequential(
            # nn.Conv2d(in_channels = 128,out_channels = 128, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 64,out_channels = 64, groups = 64, dilation  = 1,padding = 1,kernel_size= (3,3)),    # in 4, out 4, RF ?
            nn.Conv2d(in_channels = 64,out_channels = 128, dilation  = 1,padding = 0,kernel_size= (1,1)),                  # in 4, out 4, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.convblock4b = nn.Sequential(
            # nn.Conv2d(in_channels = 128,out_channels = 128, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 128,out_channels = 128, groups = 128, dilation  = 1,padding = 1,kernel_size= (3,3)),    # in 4, out 4, RF ?
            nn.Conv2d(in_channels = 128,out_channels = 256, dilation  = 1,padding = 0,kernel_size= (1,1)),                  # in 4, out 4, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )


# GAP LAYER
        self.gap = nn.AvgPool2d(4)                                                                                          # in 4, out 1, RF ?
        self.convblockf = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 


    def forward(self, x):
        x = self.convblock1a(x)
        x = self.convblock1b(x)
        x = self.convpool1(x)
        x = self.convblock2a(x)
        x = self.convblock2b(x)
        x = self.convpool2(x)
        x = self.convblock3a(x)
        x = self.convblock3b(x)
        x = self.convpool3(x)
        x = self.convblock4a(x)
        x = self.convblock4b(x)
        x = self.gap(x)
        x = self.convblockf(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)



class CifarNet7(nn.Module):
    def __init__(self):
        super(CifarNet7, self).__init__()

# FIRST MAJOR BLOCK
        # first convolutional block
        self.convblock1a = nn.Sequential(
            nn.Conv2d(in_channels = 3,out_channels = 16, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        self.convblock1b = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        # Using stride 2 without dilation instead of maxpooling
        self.convpool1 = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 1,padding = 1,kernel_size= (3,3),stride = 2), # in 32, out 16, RF ?
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 1,padding = 0,kernel_size= (1,1),stride = 1)   # to check if this can compensate for the mx pooling feature loss       
        )


# SECOND MAJOR BLOCK
          # first convolutional block
        self.convblock2a = nn.Sequential(
            # nn.Conv2d(in_channels = 16,out_channels = 32, dilation  = 1,padding = 1,kernel_size= (3,3)),                    # in 16, out 16, RF ?
            nn.Conv2d(in_channels = 16,out_channels = 16,groups = 16, dilation  = 1,padding = 1,kernel_size= (3,3)),
            nn.Conv2d(in_channels = 16,out_channels = 32, dilation = 1,padding = 0,kernel_size= (1,1)), # 8, 8, 3  
            nn.ReLU(),
            nn.BatchNorm2d(32)
        ) # will the learning will be better in depth wise or in normal 3x3 convolutions?

        self.convblock2b = nn.Sequential(
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

        # Using stride 2 without dilation to simulate max pooling
        self.convpool2 = nn.Sequential(
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 1,padding = 1,kernel_size= (3,3),stride = 2),
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 1,padding = 0,kernel_size= (1,1),stride = 1)          # in 16, out 8, RF ?
        )

# THIRD MAJOR BLOCK
         # first convolutional block
        self.convblock3a = nn.Sequential(
            # nn.Conv2d(in_channels = 32,out_channels = 64, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 32,out_channels = 32,groups = 32, dilation  = 1,padding = 1,kernel_size= (3,3)),        # in 8, out 8, RF ?
            nn.Conv2d(in_channels = 32,out_channels = 64, dilation = 1,padding = 0,kernel_size= (1,1)), # 8, 8, 3           # in 8, out 8, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.convblock3b = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 2,padding = 1, kernel_size= (3,3)),                     # in 8, out 6, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        
        # using dilation to simulate maxpooling
        self.convpool3 = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 1,padding = 1,kernel_size= (3,3),stride = 2),          # in 6, out 3, RF ?
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 1,padding = 0,kernel_size= (1,1),stride = 1)
        )

# FOURTH MAJOR BLOCK
        # second convolutional block
        self.convblock4a = nn.Sequential(
            # nn.Conv2d(in_channels = 128,out_channels = 128, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 64,out_channels = 64, groups = 64, dilation  = 1,padding = 1,kernel_size= (3,3)),    # in 3, out 3, RF ?
            nn.Conv2d(in_channels = 64,out_channels = 128, dilation  = 1,padding = 0,kernel_size= (1,1)),                  # in 3, out 3, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.convblock4b = nn.Sequential(
            # nn.Conv2d(in_channels = 128,out_channels = 128, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 128,out_channels = 128, groups = 128, dilation  = 1,padding = 1,kernel_size= (3,3)),    # in 3, out 3,, RF ?
            nn.Conv2d(in_channels = 128,out_channels = 256, dilation  = 1,padding = 0,kernel_size= (1,1)),                  # in 3, out 3, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )


# GAP LAYER
        self.gap = nn.AvgPool2d(3)                                                                                          # in 3, out 1, RF ?
        self.convblockf = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 


    def forward(self, x):
        x = self.convblock1a(x)
        x = self.convblock1b(x)
        x = self.convpool1(x)
        x = self.convblock2a(x)
        x = self.convblock2b(x)
        x = self.convpool2(x)
        x = self.convblock3a(x)
        x = self.convblock3b(x)
        x = self.convpool3(x)
        x = self.convblock4a(x)
        x = self.convblock4b(x)
        x = self.gap(x)
        x = self.convblockf(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


class CifarNet8(nn.Module):
    def __init__(self):
        super(CifarNet8, self).__init__()

# FIRST MAJOR BLOCK
        # first convolutional block
        self.convblock1a = nn.Sequential(
            nn.Conv2d(in_channels = 3,out_channels = 16, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        self.convblock1b = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        # Using stride 2 without dilation instead of maxpooling
        self.convpool1 = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 8,padding = 0,kernel_size= (3,3),stride = 1), # in 32, out 16, RF ?
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 1,padding = 0,kernel_size= (1,1),stride = 1)   # to check if this can compensate for the mx pooling feature loss       
        )


# SECOND MAJOR BLOCK
          # first convolutional block
        self.convblock2a = nn.Sequential(
            # nn.Conv2d(in_channels = 16,out_channels = 32, dilation  = 1,padding = 1,kernel_size= (3,3)),                    # in 16, out 16, RF ?
            nn.Conv2d(in_channels = 16,out_channels = 16,groups = 16, dilation  = 1,padding = 1,kernel_size= (3,3)),
            nn.Conv2d(in_channels = 16,out_channels = 32, dilation = 1,padding = 0,kernel_size= (1,1)), # 8, 8, 3  
            nn.ReLU(),
            nn.BatchNorm2d(32)
        ) # will the learning will be better in depth wise or in normal 3x3 convolutions?

        self.convblock2b = nn.Sequential(
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

        # Using stride 2 without dilation to simulate max pooling
        self.convpool2 = nn.Sequential(
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 4,padding = 0,kernel_size= (3,3),stride = 1),
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 1,padding = 0,kernel_size= (1,1),stride = 1)          # in 16, out 8, RF ?
        )

# THIRD MAJOR BLOCK
         # first convolutional block
        self.convblock3a = nn.Sequential(
            # nn.Conv2d(in_channels = 32,out_channels = 64, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 32,out_channels = 32,groups = 32, dilation  = 1,padding = 1,kernel_size= (3,3)),        # in 8, out 8, RF ?
            nn.Conv2d(in_channels = 32,out_channels = 64, dilation = 1,padding = 0,kernel_size= (1,1)), # 8, 8, 3           # in 8, out 8, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.convblock3b = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 2,padding = 2, kernel_size= (3,3)),                     # in 8, out 6, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        
        # using dilation to simulate maxpooling
        self.convpool3 = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 2,padding = 0,kernel_size= (3,3),stride = 1),          # in 6, out 3, RF ?
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 1,padding = 0,kernel_size= (1,1),stride = 1)
        )

# FOURTH MAJOR BLOCK
        # second convolutional block
        self.convblock4a = nn.Sequential(
            # nn.Conv2d(in_channels = 128,out_channels = 128, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 64,out_channels = 64, groups = 64, dilation  = 1,padding = 1,kernel_size= (3,3)),    # in 3, out 3, RF ?
            nn.Conv2d(in_channels = 64,out_channels = 128, dilation  = 1,padding = 0,kernel_size= (1,1)),                  # in 3, out 3, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.convblock4b = nn.Sequential(
            # nn.Conv2d(in_channels = 128,out_channels = 128, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 128,out_channels = 128, groups = 128, dilation  = 1,padding = 1,kernel_size= (3,3)),    # in 3, out 3,, RF ?
            nn.Conv2d(in_channels = 128,out_channels = 256, dilation  = 1,padding = 0,kernel_size= (1,1)),                  # in 3, out 3, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )


# GAP LAYER
        self.gap = nn.AvgPool2d(3)                                                                                          # in 3, out 1, RF ?
        self.convblockf = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 


    def forward(self, x):
        x = self.convblock1a(x)
        # print(x.shape)
        x = self.convblock1b(x)
        # print(x.shape)
        x = self.convpool1(x)
        # print(x.shape)
        x = self.convblock2a(x)
        # print(x.shape)
        x = self.convblock2b(x)
        # print(x.shape)
        x = self.convpool2(x)
        # print(x.shape)
        x = self.convblock3a(x)
        # print(x.shape)
        x = self.convblock3b(x)
        # print(x.shape)
        x = self.convpool3(x)
        # print(x.shape)
        x = self.convblock4a(x)
        # print(x.shape)
        x = self.convblock4b(x)
        # print(x.shape)
        x = self.gap(x)
        # print(x.shape)
        x = self.convblockf(x)
        # print(x.shape)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class CifarNet9(nn.Module):
    def __init__(self):
        super(CifarNet9, self).__init__()

# FIRST MAJOR BLOCK
        # first convolutional block
        self.convblock1a = nn.Sequential(
            nn.Conv2d(in_channels = 3,out_channels = 16, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        self.convblock1b = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        self.convblock1c = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        # Using stride 2 without dilation instead of maxpooling
        self.convpool1 = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 8,padding = 0,kernel_size= (3,3),stride = 1), # in 32, out 16, RF ?
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 1,padding = 0,kernel_size= (1,1),stride = 1)   # to check if this can compensate for the mx pooling feature loss       
        )


# SECOND MAJOR BLOCK
          # first convolutional block
        self.convblock2a = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels = 32, dilation  = 1,padding = 1,kernel_size= (3,3)),                    # in 16, out 16, RF ?
            # nn.Conv2d(in_channels = 16,out_channels = 16,groups = 16, dilation  = 1,padding = 1,kernel_size= (3,3)),
            # nn.Conv2d(in_channels = 16,out_channels = 32, dilation = 1,padding = 0,kernel_size= (1,1)), # 8, 8, 3  
            nn.ReLU(),
            nn.BatchNorm2d(32)
        ) # will the learning will be better in depth wise or in normal 3x3 convolutions?

        self.convblock2b = nn.Sequential(
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 2,padding = 2, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        self.convblock2c = nn.Sequential(
            nn.Conv2d(in_channels = 32,out_channels = 32,groups = 32, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 1,padding = 0, kernel_size= (1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

        # Using stride 2 without dilation to simulate max pooling
        self.convpool2 = nn.Sequential(
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 4,padding = 0,kernel_size= (3,3),stride = 1),
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 1,padding = 0,kernel_size= (1,1),stride = 1)          # in 16, out 8, RF ?
        )

# THIRD MAJOR BLOCK
         # first convolutional block
        self.convblock3a = nn.Sequential(
            # nn.Conv2d(in_channels = 32,out_channels = 64, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 32,out_channels = 32,groups = 32, dilation  = 1,padding = 1,kernel_size= (3,3)),        # in 8, out 8, RF ?
            nn.Conv2d(in_channels = 32,out_channels = 64, dilation = 1,padding = 0,kernel_size= (1,1)), # 8, 8, 3           # in 8, out 8, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.convblock3b = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 8, out 6, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.convblock3c = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64, groups = 1,dilation  = 1,padding = 1, kernel_size= (3,3)), 
            # nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 1,padding = 0, kernel_size= (1,1)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        
        # using dilation to simulate maxpooling
        self.convpool3 = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 2,padding = 0,kernel_size= (3,3),stride = 1),          # in 6, out 3, RF ?
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 1,padding = 0,kernel_size= (1,1),stride = 1)
        )

# FOURTH MAJOR BLOCK
        # second convolutional block
        self.convblock4a = nn.Sequential(
            # nn.Conv2d(in_channels = 128,out_channels = 128, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 64,out_channels = 64, groups = 64, dilation  = 1,padding = 1,kernel_size= (3,3)),    # in 3, out 3, RF ?
            nn.Conv2d(in_channels = 64,out_channels = 128, dilation  = 1,padding = 0,kernel_size= (1,1)),                  # in 3, out 3, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.convblock4b = nn.Sequential(
            # nn.Conv2d(in_channels = 128,out_channels = 128, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 128,out_channels = 128, groups = 128, dilation  = 1,padding = 1,kernel_size= (3,3)),    # in 3, out 3,, RF ?
            nn.Conv2d(in_channels = 128,out_channels = 256, dilation  = 1,padding = 0,kernel_size= (1,1)),                  # in 3, out 3, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )


# GAP LAYER
        self.gap = nn.AvgPool2d(3)                                                                                          # in 3, out 1, RF ?
        self.convblockf = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 


    def forward(self, x):
        x = self.convblock1a(x)
        # print(x.shape)
        x = self.convblock1b(x)
        x = self.convblock1c(x)
        # print(x.shape)
        x = self.convpool1(x)
        # print(x.shape)
        x = self.convblock2a(x)
        # print(x.shape)
        x1 = self.convblock2b(x)
        x = torch.add(x1, x1) # 
        x = self.convblock2c(x)
        # print(x.shape)
        x = self.convpool2(x)
        # print(x.shape)
        x = self.convblock3a(x)
        # print(x.shape)
        x = self.convblock3b(x)
        x = self.convblock3c(x)
        # print(x.shape)
        x = self.convpool3(x)
        # print(x.shape)
        x = self.convblock4a(x)
        # print(x.shape)
        x = self.convblock4b(x)
        # print(x.shape)
        x = self.gap(x)
        # print(x.shape)
        x = self.convblockf(x)
        # print(x.shape)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


class CifarNet10(nn.Module):
    def __init__(self):
        super(CifarNet10, self).__init__()

# FIRST MAJOR BLOCK
        # first convolutional block
        self.convblock1a = nn.Sequential(
            nn.Conv2d(in_channels = 3,out_channels = 16, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        self.convblock1b = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        # Using stride 2 without dilation instead of maxpooling
        self.convpool1 = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 1,padding = 1,kernel_size= (3,3),stride = 2), # in 32, out 16, RF ?
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 1,padding = 0,kernel_size= (1,1),stride = 1)   # to check if this can compensate for the mx pooling feature loss       
        )


# SECOND MAJOR BLOCK
          # first convolutional block
        self.convblock2a = nn.Sequential(
            # nn.Conv2d(in_channels = 16,out_channels = 32, dilation  = 1,padding = 1,kernel_size= (3,3)),                    # in 16, out 16, RF ?
            nn.Conv2d(in_channels = 16,out_channels = 16,groups = 16, dilation  = 1,padding = 1,kernel_size= (3,3)),
            nn.Conv2d(in_channels = 16,out_channels = 32, dilation = 1,padding = 0,kernel_size= (1,1)), # 8, 8, 3  
            nn.ReLU(),
            nn.BatchNorm2d(32)
        ) # will the learning will be better in depth wise or in normal 3x3 convolutions?

        self.convblock2b = nn.Sequential(
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 16, out 16, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

        # Using stride 2 without dilation to simulate max pooling
        self.convpool2 = nn.Sequential(
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 1,padding = 1,kernel_size= (3,3),stride = 2),
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 1,padding = 0,kernel_size= (1,1),stride = 1)          # in 16, out 8, RF ?
        )

# THIRD MAJOR BLOCK
         # first convolutional block
        self.convblock3a = nn.Sequential(
            # nn.Conv2d(in_channels = 32,out_channels = 64, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 32,out_channels = 32,groups = 32, dilation  = 1,padding = 1,kernel_size= (3,3)),        # in 8, out 8, RF ?
            nn.Conv2d(in_channels = 32,out_channels = 64, dilation = 1,padding = 0,kernel_size= (1,1)), # 8, 8, 3           # in 8, out 8, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.convblock3b = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 2,padding = 2, kernel_size= (3,3)),                     # in 8, out 8, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        
        # using dilation to simulate maxpooling
        self.convpool3 = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 1,padding = 1,kernel_size= (3,3),stride = 2),          # in 8, out 4, RF ?
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 1,padding = 0,kernel_size= (1,1),stride = 1)
        )

# FOURTH MAJOR BLOCK
        # second convolutional block
        self.convblock4a = nn.Sequential(
            # nn.Conv2d(in_channels = 128,out_channels = 128, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 64,out_channels = 64, groups = 64, dilation  = 1,padding = 1,kernel_size= (3,3)),    # in 4, out 4, RF ?
            nn.Conv2d(in_channels = 64,out_channels = 128, dilation  = 1,padding = 0,kernel_size= (1,1)),                  # in 4, out 4, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.convblock4b = nn.Sequential(
            # nn.Conv2d(in_channels = 128,out_channels = 128, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 128,out_channels = 128, groups = 128, dilation  = 1,padding = 1,kernel_size= (3,3)),    # in 4, out 4,, RF ?
            nn.Conv2d(in_channels = 128,out_channels = 256, dilation  = 1,padding = 0,kernel_size= (1,1)),                  # in 4, out 4, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )


# GAP LAYER
        self.gap = nn.AvgPool2d(4)                                                                                          # in 3, out 1, RF ?
        self.convblockf = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 


    def forward(self, x):
        x = self.convblock1a(x)
        x = self.convblock1b(x)
        x = self.convpool1(x)
        x = self.convblock2a(x)
        x = self.convblock2b(x)
        x = self.convpool2(x)
        x = self.convblock3a(x)
        x1 = self.convblock3b(x)
        x = torch.add(x,x1)
        x = self.convpool3(x)
        x = self.convblock4a(x)
        x = self.convblock4b(x)
        x = self.gap(x)
        x = self.convblockf(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


class CifarNet11(nn.Module):
    def __init__(self):
        super(CifarNet11, self).__init__()

# FIRST MAJOR BLOCK
        # first convolutional block
        self.convblock1a = nn.Sequential(
            nn.Conv2d(in_channels = 3,out_channels = 16, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        self.convblock1b = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        # Using stride 2 without dilation instead of maxpooling
        self.convpool1 = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 1,padding = 1,kernel_size= (3,3),stride = 2), # in 32, out 16, RF ?
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 1,padding = 0,kernel_size= (1,1),stride = 1)   # to check if this can compensate for the mx pooling feature loss       
        )


# SECOND MAJOR BLOCK
          # first convolutional block
        self.convblock2a = nn.Sequential(
            # nn.Conv2d(in_channels = 16,out_channels = 32, dilation  = 1,padding = 1,kernel_size= (3,3)),                    # in 16, out 16, RF ?
            nn.Conv2d(in_channels = 16,out_channels = 16,groups = 16, dilation  = 1,padding = 1,kernel_size= (3,3)),
            nn.Conv2d(in_channels = 16,out_channels = 32, dilation = 1,padding = 0,kernel_size= (1,1)), # 8, 8, 3  
            nn.ReLU(),
            nn.BatchNorm2d(32)
        ) # will the learning will be better in depth wise or in normal 3x3 convolutions?

        self.convblock2b = nn.Sequential(
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 16, out 16, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

        # Using stride 2 without dilation to simulate max pooling
        self.convpool2 = nn.Sequential(
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 1,padding = 1,kernel_size= (3,3),stride = 2),
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 1,padding = 0,kernel_size= (1,1),stride = 1)          # in 16, out 8, RF ?
        )

# THIRD MAJOR BLOCK
         # first convolutional block
        self.convblock3a = nn.Sequential(
            # nn.Conv2d(in_channels = 32,out_channels = 64, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 32,out_channels = 32,groups = 32, dilation  = 1,padding = 1,kernel_size= (3,3)),        # in 8, out 8, RF ?
            nn.Conv2d(in_channels = 32,out_channels = 64, dilation = 1,padding = 0,kernel_size= (1,1)), # 8, 8, 3           # in 8, out 8, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.convblock3b = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 2,padding = 2, kernel_size= (3,3)),                     # in 8, out 8, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        
        # using dilation to simulate maxpooling
        self.convpool3 = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 1,padding = 1,kernel_size= (3,3),stride = 2),          # in 8, out 4, RF ?
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 1,padding = 0,kernel_size= (1,1),stride = 1)
        )

# FOURTH MAJOR BLOCK
        # second convolutional block
        self.convblock4a = nn.Sequential(
            # nn.Conv2d(in_channels = 128,out_channels = 128, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 64,out_channels = 64, groups = 64, dilation  = 1,padding = 1,kernel_size= (3,3)),    # in 4, out 4, RF ?
            nn.Conv2d(in_channels = 64,out_channels = 128, dilation  = 1,padding = 0,kernel_size= (1,1)),                  # in 4, out 4, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.convblock4b = nn.Sequential(
            # nn.Conv2d(in_channels = 128,out_channels = 128, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 128,out_channels = 128, groups = 128, dilation  = 1,padding = 1,kernel_size= (3,3)),    # in 4, out 4,, RF ?
            nn.Conv2d(in_channels = 128,out_channels = 256, dilation  = 1,padding = 0,kernel_size= (1,1)),                  # in 4, out 4, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )


# GAP LAYER
        self.gap = nn.AvgPool2d(4)                                                                                          # in 3, out 1, RF ?
        self.convblockf = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 


    def forward(self, x):
        x = self.convblock1a(x)
        x1 = self.convblock1b(x)
        x = torch.add(x,x1)
        x = self.convpool1(x)


        x = self.convblock2a(x)
        x1 = self.convblock2b(x)
        x = torch.add(x,x1)
        x = self.convpool2(x1)

        x = self.convblock3a(x)
        x1 = self.convblock3b(x)        
        x = torch.add(x,x1)
        x = self.convpool3(x1)

        x = self.convblock4a(x)
        x = self.convblock4b(x)
        
        x = self.gap(x)
        x = self.convblockf(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class CifarNet12(nn.Module):
    def __init__(self):
        super(CifarNet12, self).__init__()

# FIRST MAJOR BLOCK
        # first convolutional block
        self.convblock1a = nn.Sequential(
            nn.Conv2d(in_channels = 3,out_channels = 16, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        self.convblock1b = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels = 16,groups =16, dilation  = 1,padding = 1, kernel_size= (3,3)),  
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 1,padding = 0, kernel_size= (1,1)),                     # in 32, out 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        # Using stride 2 without dilation instead of maxpooling
        self.convpool1 = nn.Sequential(
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 1,padding = 1,kernel_size= (3,3),stride = 2), # in 32, out 16, RF ?
            nn.Conv2d(in_channels = 16,out_channels = 16, dilation  = 1,padding = 0,kernel_size= (1,1),stride = 1)   # to check if this can compensate for the mx pooling feature loss       
        )


# SECOND MAJOR BLOCK
          # first convolutional block
        self.convblock2a = nn.Sequential(
            # nn.Conv2d(in_channels = 16,out_channels = 32, dilation  = 1,padding = 1,kernel_size= (3,3)),                    # in 16, out 16, RF ?
            nn.Conv2d(in_channels = 16,out_channels = 16,groups = 16, dilation  = 1,padding = 1,kernel_size= (3,3)),
            nn.Conv2d(in_channels = 16,out_channels = 32, dilation = 1,padding = 0,kernel_size= (1,1)), # 8, 8, 3  
            nn.ReLU(),
            nn.BatchNorm2d(32)
        ) # will the learning will be better in depth wise or in normal 3x3 convolutions?

        self.convblock2b = nn.Sequential(
            nn.Conv2d(in_channels = 32,out_channels = 32, groups = 32, dilation  = 1,padding = 1, kernel_size= (3,3)),                     # in 16, out 16, RF 3
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 1,padding = 0, kernel_size= (1,1)), 
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

        # Using stride 2 without dilation to simulate max pooling
        self.convpool2 = nn.Sequential(
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 1,padding = 1,kernel_size= (3,3),stride = 2),
            nn.Conv2d(in_channels = 32,out_channels = 32, dilation  = 1,padding = 0,kernel_size= (1,1),stride = 1)          # in 16, out 8, RF ?
        )

# THIRD MAJOR BLOCK
         # first convolutional block
        self.convblock3a = nn.Sequential(
            # nn.Conv2d(in_channels = 32,out_channels = 64, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 32,out_channels = 32,groups = 32, dilation  = 1,padding = 1,kernel_size= (3,3)),        # in 8, out 8, RF ?
            nn.Conv2d(in_channels = 32,out_channels = 64, dilation = 1,padding = 0,kernel_size= (1,1)), # 8, 8, 3           # in 8, out 8, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.convblock3b = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64,groups = 64, dilation  = 2,padding = 2, kernel_size= (3,3)),                     # in 8, out 8, RF 3
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 1,padding = 0, kernel_size= (1,1)), 
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        
        # using dilation to simulate maxpooling
        self.convpool3 = nn.Sequential(
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 1,padding = 1,kernel_size= (3,3),stride = 2),          # in 8, out 4, RF ?
            nn.Conv2d(in_channels = 64,out_channels = 64, dilation  = 1,padding = 0,kernel_size= (1,1),stride = 1)
        )

# FOURTH MAJOR BLOCK
        # second convolutional block
        self.convblock4a = nn.Sequential(
            # nn.Conv2d(in_channels = 128,out_channels = 128, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 64,out_channels = 64, groups = 64, dilation  = 1,padding = 1,kernel_size= (3,3)),    # in 4, out 4, RF ?
            nn.Conv2d(in_channels = 64,out_channels = 128, dilation  = 1,padding = 0,kernel_size= (1,1)),                  # in 4, out 4, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.convblock4b = nn.Sequential(
            # nn.Conv2d(in_channels = 128,out_channels = 128, dilation  = 1,padding = 1,kernel_size= (3,3)), 
            nn.Conv2d(in_channels = 128,out_channels = 128, groups = 128, dilation  = 1,padding = 1,kernel_size= (3,3)),    # in 4, out 4,, RF ?
            nn.Conv2d(in_channels = 128,out_channels = 176, dilation  = 1,padding = 0,kernel_size= (1,1)),                  # in 4, out 4, RF ?
            nn.ReLU(),
            nn.BatchNorm2d(176)
        )

# GAP LAYER
        self.gap = nn.AvgPool2d(4)                                                                                          # in 3, out 1, RF ?
        self.convblockf = nn.Sequential(
            nn.Conv2d(in_channels=176, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 


    def forward(self, x):
        x = self.convblock1a(x)
        x1 = self.convblock1b(x)
        x = torch.mul(x,x1)
        x = self.convpool1(x)


        x = self.convblock2a(x)
        x1 = self.convblock2b(x)
        x = torch.mul(x,x1)
        x = self.convpool2(x)

        x = self.convblock3a(x)
        x1 = self.convblock3b(x)
        x = torch.mul(x,x1)
        x = self.convpool3(x)

        x = self.convblock4a(x)
        x = self.convblock4b(x)
        
        x = self.gap(x)
        x = self.convblockf(x)
        x = x.view(-1, 10)
        
        return F.log_softmax(x, dim=-1)