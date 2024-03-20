import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader,random_split
from torchvision import transforms as T
import matplotlib.pyplot as plt
import torch



class CustomData(Dataset):
    def __init__(self, root, transformation=None):
        """
        Custom dataset class for loading images and labels from a CSV file.

        Args:
            root (str): Path to the CSV file containing image paths and labels.
            transformation : Optional transformation to be applied to the images.
        """
        self.transformation = transformation
        df = pd.read_csv(root)
        self.images = []
        self.labels = []
        self.classes = {}
        columns = df.loc[:, df.columns.drop('label')]
        labels = df['label']
        
        class_counter = 0
        for idx, (image, label) in enumerate(zip(columns.values, labels)):
            im_shapes = np.reshape(image, (28, 28, 3))
            self.images.append(im_shapes)
            
            if label not in self.classes:
                self.classes[label] = class_counter
                class_counter += 1
                
            self.labels.append(self.classes[label])
        
    def __len__(self):
        """
        Returns the number of images in the dataset.
        """
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding label from the dataset.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: Tuple containing the image and its label.
        """
        im = Image.fromarray(self.images[idx].astype("uint8"), "RGB")
        gt = self.labels[idx]
        
        if self.transformation:
            im = self.transformation(im)
            
        return im, gt


# mean, std = [0.2250, 0.2250, 0.2250], [0.2505, 0.2505, 0.2505]
# tfs = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean = mean, std = std)])
# root = "D:/Data/Datasets/Skin_lesion_Pixel_data/meta_deta.csv"
# ds = CustomData(root, transformation=tfs)

# print(ds[0][0].shape)