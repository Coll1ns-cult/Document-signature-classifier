import os 
import pandas as pd
import torch 
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
import numpy as np

class SignatureDataset(Dataset):
    def __init__(self, label_file, root_dir, transform = None):
        '''Arguments:
        label_file: path to csv file which contains 2 columns:
        one with name of the person, other with the label
        root_dir: path file of images
        transform: transformations that will be applied (default: None)'''
        self.df = pd.read_csv(label_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        '''lenght of the file'''
        return len(self.df)
    def __getitem__(self, index): #-> (List[torch.tensor], torch.tensor)
        '''getting image based on index (protocol in MapDatasets)'''
        images = []
        labels = []
        for ind in index: #since our index coming from 
            img_path = os.path.join(self.root_dir, str(self.df.iloc[ind, 0]))
            image_i = io.imread(img_path)
            label_i = torch.tensor(int(self.df.iloc[ind, 1]))

            if self.transform:
                '''transformations to be done to image'''
                image_i = self.transform(image_i)
            images.append(image_i)
            labels.append(label_i)

        #added squeeze to test it
        label = torch.stack(labels)
        return (images, label)
