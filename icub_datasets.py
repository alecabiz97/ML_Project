import os
import torch.utils.data
from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np

class ICubWorld7(torch.utils.data.Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None):
        if train:
            self.root = os.path.join(root, 'train')
        else:
            self.root = os.path.join(root, 'test')

        # self.img_info is a numpy array:
        # 1° column -> file directory
        # 2° column -> label
        self.img_info = []
        for label in os.listdir(self.root):
            class_dir = os.path.join(self.root, label)
            for fname in os.listdir(class_dir):
                self.img_info.append((os.path.join(label, fname), label))
        # Convert to numpy array
        self.img_info = np.array(self.img_info)
        self.labels = np.unique(self.img_info[:, 1])
        # Create a dictionary class_to_idx to convert string labels to integer
        self.class_to_idx = {lab: num for num, lab in enumerate(self.labels)}
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        ''' Return the number of samples in the dataset'''
        return len(self.img_info)

    def __getitem__(self, idx):
        '''Return the image and the label of the element of index idx
            :param
                idx: integer
            :return
                image: Tensor(3,80,80)
                label: Integer
            '''
        img_path = os.path.join(self.root, self.img_info[idx, 0])
        image = Image.open(img_path)
        label = self.img_info[idx, 1]  # the label is a string
        label = self.class_to_idx[label]  # the label is an integer
        if self.transform:
            image = self.transform(image)  # apply a transformation (ex ToTensor())
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def get_data(self):
        '''Return all the sample
            :return
                X: Tensor(n_smaples,n_channel,height,width) All the data
                y: Tensor(n_samples) All the labels
        '''
        trans = ToTensor()
        img, label = self.__getitem__(0)
        if self.transform is None:
            img = trans(img)
        n_channel, height, width = img.shape  # dimensions of one sample
        X = torch.zeros((len(self), n_channel, height, width))
        y = torch.zeros((len(self)))
        for i in range(len(self)):
            img, label = self.__getitem__(i)
            if self.transform is None:
                img = trans(img)
            X[i, :, :, :] = img
            y[i] = label
        return X, y


class ICubWorld28(torch.utils.data.Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None,sub_label=False):
        self.sub_label=sub_label
        if train:
            self.root = os.path.join(root, 'train')
        else:
            self.root = os.path.join(root, 'test')

        # self.img_info is a numpy array:
        # 1° column -> file directory
        # 2° column -> label (es cup)
        # 3° column -> sub_label (es cup1)
        # 4° column -> day
        self.img_info = []
        for day in os.listdir(self.root):
            s = os.path.join(self.root, day)
            for label in os.listdir(s):
                # print(label)
                s2 = os.path.join(s, label)
                for sub_label in os.listdir(s2):
                    s3 = os.path.join(s2, sub_label)
                    for fname in os.listdir(s3):
                        self.img_info.append((os.path.join(day, label, sub_label, fname), label, sub_label, day))

        # Convert to numpy array
        self.img_info = np.array(self.img_info)
        if self.sub_label:
            self.labels = np.unique(self.img_info[:, 2]) # ex cup1
        else:
            self.labels = np.unique(self.img_info[:, 1]) # ex cup
        # Create a dictionary class_to_idx to convert string labels to integer
        self.class_to_idx = {lab: num for num, lab in enumerate(self.labels)}
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        ''' Return the number of samples in the dataset'''
        return self.img_info.shape[0]

    def __getitem__(self, idx):
        '''Return the image and the label of the element of index idx
            :param
                idx: integer
            :return
                image: Tensor(3,240,320)
                label: Integer
            '''
        img_path = os.path.join(self.root, self.img_info[idx, 0])
        image = Image.open(img_path)
        if self.sub_label:
            label = self.img_info[idx, 2]  # the label is a string
        else:
            label = self.img_info[idx, 1]
        label = self.class_to_idx[label]  # the label is an integer
        if self.transform:
            image = self.transform(image)  # apply a transformation (es ToTensor())
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def get_data(self):
        '''Return all the sample
            :return
                X: Tensor(n_samples,n_channel,height,width) All the data
                y: Tensor(n_samples) All the labels
        '''
        trans = ToTensor()
        img, label = self.__getitem__(0)
        if self.transform is None:
            img = trans(img)
        n_channel, height, width = img.shape  # dimensions of one sample
        X = torch.zeros((len(self), n_channel, height, width))
        y = torch.zeros((len(self)))
        for i in range(len(self)):
            img, label = self.__getitem__(i)
            if self.transform is None:
                img = trans(img)
            X[i, :, :, :] = img
            y[i] = label
        return X, y


if __name__ == '__main__':
    # Prova ICubWorld7
    # root = "..\\iCubWorld1.0\\human"
    # training_data = ICubWorld7(root)
    # img, y = training_data.__getitem__(1)
    # # imshow(img)
    # # print(len(training_data))
    # # print(len(test_data))
    # X, y = training_data.get_data()

    # Prova ICubWorld28
    root = 'ICubWorld28'
    d = ICubWorld28(root,sub_label=True)
    print(len(d.labels))
    img, y = d.__getitem__(200)
    print(y)
    img.show()
    print(d.labels[y])
    # X, y = d.get_data()
