import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # Non so cosa significa, ma lo cercherò, lo troverò e lo capirò
import numpy as np
import torch.utils.data
from matplotlib import pyplot as plt
from utils import *
from PIL import Image
from torchvision.transforms import ToTensor


class ICubDataset(torch.utils.data.Dataset):

    def __init__(self, root,train=True, transform=None, target_transform=None):
        if train:
            self.root = os.path.join(root,'train')
        else:
            self.root=os.path.join(root,'test')

        # self.img_labels is a numpy array, in the first column there is the final part of the directory
        # and in the second one the label like [[bottle//0000000.ppm, 'bottle'],...]
        self.img_labels=[]
        for label in os.listdir(self.root):
            class_dir = os.path.join(self.root, label)
            for fname in os.listdir(class_dir):
                self.img_labels.append((os.path.join(label, fname),label))
        # Convert to numpy array
        self.img_labels=np.array(self.img_labels)
        # Create a dictionary class_to_idx to convert string labels to integer
        labels = np.unique(self.img_labels[:, 1])
        self.class_to_idx = {lab: num for num, lab in enumerate(labels)}
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        ''' Return the number of samples in the dataset'''
        return len(self.img_labels)

    def __getitem__(self, idx):
        '''Return the image and the label of the element of index idx
            :param
                idx: integer
            :return
                image: Tensor(3,80,80)
                label: Integer
            '''
        img_path = os.path.join(self.root, self.img_labels[idx, 0])
        image = Image.open(img_path)
        label = self.img_labels[idx, 1] # the label is a string
        label=self.class_to_idx[label] # the label is an integer
        if self.transform:
            image = self.transform(image) # apply a transformation (ex ToTensor())
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def get_data(self):
        '''Return all the sample
            :return
                X: Tensor(n_smaples,n_channel,height,width) All the data
                y: Tensor(n_samples) All the labels
        '''
        img, label = self.__getitem__(0)
        n_channel,height,width=img.shape # dimensions of one sample
        X=torch.zeros((len(self),n_channel,height,width))
        y=torch.zeros((len(self)))
        for i in range(len(self)):
            img,label=self.__getitem__(i)
            X[i,:,:,:]=img
            y[i]=label
        return X,y


if __name__=='__main__':
    root="..\\iCubWorld1.0\\human"

    #createFileCsv('robot_train.csv',os.path.join(root,'train'))
    #createFileCsv('robot_test.csv', os.path.join(root,'test'))

    training_data=ICubDataset(root,
        train=True,
        transform=ToTensor())

    test_data = ICubDataset(root,
                            train=False,
                            transform=ToTensor())

    img,y=training_data.__getitem__(1)
    #imshow(img)
    print(len(training_data))
    print(len(test_data))

    X,y=training_data.get_data()