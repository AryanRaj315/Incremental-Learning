from torchvision.datasets import CIFAR10
import numpy as np
import torch
from PIL import Image
import pandas as pd
import cv2

class mnist():
    def __init__(self, root = '../',
                 classes = range(10), train = True, 
                 transform = None, target_transform = None,
                 download = False):
        
        self.classes = classes
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        train = pd.read_csv("train.csv")
        test = pd.read_csv("test.csv")
        self.targets = train["label"]
        X_train = train.drop(labels = ["label"],axis = 1) 
        X_train = X_train / 255.0
        test = test / 255.0
        # Reshaping for images
        self.images = X_train.values.reshape(-1,28,28,1)
        self.test = test.values.reshape(-1,28,28,1)
        if self.train:
            train_data = []
            train_labels = []

            for i in range(len(self.images)):
                if self.targets[i] in classes:
                    gray = self.images[i][:,:,0].astype(np.float32)
                    img = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
                    train_data.append(img)
                    train_labels.append(self.targets[i])

            self.train_data = np.array(train_data)
            self.train_labels = train_labels

        else:
            test_data = []
            test_labels = []

            for i in range(len(self.test)):
                if self.targets[i] in classes:
                    gray = self.images[i][:,:,0].astype(np.float32)
                    img = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
                    test_data.append(img)
                    test_labels.append(self.targets[i])

            self.test_data = test_data
            self.test_labels = test_labels
   

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
    
    def get_image_class(self, label):
        return self.train_data[np.array(self.train_labels) == label]

    def append(self, images, labels):
        """Append dataset with images and labels

        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """

        self.train_data = np.concatenate((self.train_data, images), axis=0)
        self.train_labels = self.train_labels + labels

    

