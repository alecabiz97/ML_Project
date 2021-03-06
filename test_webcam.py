import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image
import pickle
from icub_datasets import ICubWorld28
from copy import copy

root = 'ICubWorld28'
dataset = ICubWorld28(root, train=True)
labels = dataset.labels

# Loading the model chosen for real-time classification
f = open('trained_models/squeezenet1_0_0.01_7_classes.pkl', 'rb')
model = pickle.load(f)
f.close()

# Use the GPU if available
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Image Transformation and Normalization
data_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Video Acquisition
cap = cv2.VideoCapture(0)
while 1:
    ret, frame = cap.read()

    if ret == False:
        print('The webcam is not available')
        break

    height, width = int(cap.get(3)), int(cap.get(4))

    frame2 = copy(frame)

    # Show a rectangle to mark the cropped area for recognition
    cv2.rectangle(frame2, (int(height / 2) - 112, int(width / 2) - 112), (int(height / 2) + 112, int(width / 2) + 112),
                  (0, 255, 0), 4)
    img = frame[int(width / 2) - 112:int(width / 2) + 112, int(height / 2) - 112:int(height / 2) + 112]

    cv2.imshow('img', img)
    img = Image.fromarray(img)
    img = data_transforms(img)

    img = img.to(device)

    # Classification
    output = model(img.unsqueeze(0))
    score, pred = torch.max(output, 1)
    label = labels[pred.item()]

    # Put text on the upper part of the rectangle to show the label
    # with the corresponding recognition percentage
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = '{}: {:.2f}%'.format(label, score.item())
    frame2 = cv2.putText(frame2, text, (int(height / 2) - 112, int(width / 2) - 120), font, 0.7, (0, 0, 0), 2,
                         cv2.LINE_AA)
    frame2 = cv2.putText(frame2, text, (int(height / 2) - 112, int(width / 2) - 120), font, 0.7, (0, 255, 0), 1,
                         cv2.LINE_AA)

    cv2.imshow('frame', frame2)

    # Stop the video acquisition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
