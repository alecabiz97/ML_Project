from decimal import Decimal

from icub_datasets import *
import os
import numpy as np
import pandas as pd
import torch.utils.data
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
import csv
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision import transforms, models
import torch.nn.functional as F
from sklearn.svm import SVC
from utils import *

root="..\\iCubWorld1.0\\human"

# Create training_data and test_data
training_data=ICubWorld7(root,
    train=True,
    transform=ToTensor())

test_data = ICubWorld7(root,
                        train=False,
                        transform=ToTensor())

img,y=training_data.__getitem__(1)
#imshow(img)
print(len(training_data))
print(len(test_data))

# Hyper-parameters
input_size = 3* 80 * 80  # 19200
num_classes = 7
batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# # Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# PROVA SVC
clf=SVC(kernel='rbf')

# TRAINING
X_tr,y_tr=training_data.get_data()

# Reshape X_tr from (n_sample,n_channel,height,width) to (n_sample,n_channel*height*width)
n_sample,n_channel,height,width=X_tr.shape
X_tr=X_tr.view(-1,n_channel*height*width)

# Convert to numpy
X_tr,y_tr=X_tr.numpy(),y_tr.numpy()

print('X_tr shape: ',X_tr.shape)
print('y_tr: ',y_tr.shape)
clf.fit(X_tr, y_tr)

# TEST
X_ts,y_ts=test_data.get_data()

# Reshape X_tr from (n_sample,n_channel,height,width) to (n_sample,n_channel*height*width)
n_sample,n_channel,height,width=X_ts.shape
X_ts=X_ts.view(-1,n_channel*height*width)

# Convert to numpy
X_ts, y_ts = X_ts.numpy(), y_ts.numpy()


print('X_ts shape: ', X_ts.shape)
y_pred = clf.predict(X_ts)
print('Classification accuracy: ', np.mean(y_pred == y_ts))


# PROVA IMAGENET
# # Download and cache pretrained model from PyTorch model zoo
# model = models.resnet18(pretrained=True)
# # Set the model in evaluation mode (e.g., disable dropout)
# model.eval()
#
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor()
# ])
#
# normalize = transforms.Normalize(
#     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#
# # we need this later for bringing the image back to input space
# inv_normalize = transforms.Normalize(
#     mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
#     std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
#
# for input_batch,labels in train_dataloader:
#     print('INPUT BATCH: ', input_batch.shape)
#     # move the input and model to GPU for speed if available
#     if torch.cuda.is_available():
#         input_batch = input_batch.to(device)
#         model.to(device)
#
#     with torch.no_grad():
#         output = model(input_batch)
#
#     # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
#     print("Output shape", output.shape)
#
#     # softmax will rescale outputs so that the sum is 1 and we
#     # can use them as probability scores
#     scores = torch.softmax(output, dim=1)
#
#     # take top k predictions - accuracy is usually measured with top-5
#     _, preds = output.topk(k=5, dim=1)
#
#
#     # use the output as index for the labels list
#     for label in preds[0]:
#         predicted_label = labels[label.item()]
#         score = scores[0, label.item()].item()
#         print("Label: {:25s} Score: {:.2f}".format(predicted_label, Decimal(score)))
#
#     break

