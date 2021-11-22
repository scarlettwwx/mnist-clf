from os import name
import pickle
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
import cv2
import sys


class TrainData(data.Dataset):

    def __init__(self):
        with open('comp551/images_l.pkl', 'rb') as f:
            self.data_train = pickle.load(f)

        with open('comp551/labels_l.pkl', 'rb') as f:
            self.data_label = pickle.load(f)
        
        self.HW = 56

        print(self.data_train.shape)
        print(self.data_label.shape)

    def __getitem__(self, index):

        # get one item
        data = self.data_train[index] # [56, 56]
        label = self.data_label[index] # [36, ]

        #cv2.imwrite('Original image.png', data)

        # print(data.shape)
        # print(label.shape)
        
        # Augmentation
        # flip (randomly)
        if np.random.rand(1)>0.5:
            data = cv2.flip(data, 0)
               
        if np.random.rand(1)>0.5:
            data = cv2.flip(data, 1)

        #cv2.imwrite('Fliped image.png', data)
        
        # rotate
        if np.random.rand(1)>0.5:
            M_2 = cv2.getRotationMatrix2D((28, 28), -90, 1)
            data = cv2.warpAffine(data, M_2, (56, 56))
        
        #cv2.imwrite('Rotated_-90.png', data)

        # denoise
        data = cv2.GaussianBlur(data,(5,5),1)

        #cv2.imwrite('Gaussian.png', data)
        
        # to Tenser
        data = data.reshape(-1, self.HW, self.HW) # [1, 56, 56]
        label = label.reshape(-1, 36) # [1, 36]
        data = torch.from_numpy(data.astype(np.float32) / 255.0)

        return data, label

    def __len__(self):
        return self.data_train.shape[0]

class TestData(data.Dataset):

    def __init__(self):
        with open('comp551/images_test.pkl', 'rb') as f:
            self.data_test = pickle.load(f)

        print(self.data_test.shape)

    def __getitem__(self, index):

        # get one item
        data = self.data_test[index] # [1, 56, 56]

       # denoise
        data = cv2.GaussianBlur(data,(5,5),1)

        #cv2.imwrite('Gaussian.png', data)
        
        # to Tenser
        data = data.reshape(-1, self.HW, self.HW) # [1, 56, 56]
        data = torch.from_numpy(data.astype(np.float32) / 255.0)

        return data

    def __len__(self):
        return self.data_test.shape[0]
