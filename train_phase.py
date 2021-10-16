from icub_datasets import ICubWorld28
from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Generator
import numpy as np
import torchvision
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pickle
from tqdm import tqdm


def train_model(model, dataloaders, criterion, optimizer, num_epochs=50, is_inception=False):
    since = time.time()

    history = {'train': [], 'val': []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0  # Initialized loss
            running_corrects = 0  # Initialized correct checks

            # Iterate over data
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)

            # Loss and Accuracy Evaluation
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            history[phase].append((epoch_loss, epoch_acc.cpu().item()))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history


# Finetuning only on the last layers of the pretrained models
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# Initialize the model based on the chosen one
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet152":
        """ Resnet152
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet1_0":
        """ Squeezenet1.0
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "squeezenet1_1":
        """ Squeezenet1.1
        """
        model_ft = models.squeezenet1_1(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet161":
        """ Densenet
        """
        model_ft = models.densenet161(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


if __name__ == '__main__':

    # Hyperparameters
    root = 'ICubWorld28'
    batch_size = 64  # Batch size for training (change depending on how much memory you have)
    num_epochs = 50  # Number of epochs to train for
    SAVE = True  # Boolean parameter to decide if the model needs to be saved (True=Yes, False=No)
    feature_extract = True  # If True Modify only last layers of each model
    sub_label = True  # If True uses 28 classes during training otherwise it uses 7 classes
    num_classes = 28 if sub_label else 7  # Number of classes in the dataset
    SUBSET = False  # If True pick only part of the dataset

    for model_name in ['alexnet', 'squeezenet1_0', 'squeezenet1_1', 'resnet18', 'resnet152', 'densenet161']:
        for learning_rate in [0.01, 0.001, 0.0001]:

            model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

            data_transforms = transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

            print("Initializing Datasets and Dataloaders...")

            dataset = ICubWorld28(root, train=True, transform=data_transforms, sub_label=sub_label)

            # Create subset to speed up the process
            if SUBSET:
                rand_indx = np.random.permutation(np.arange(len(dataset)))[0:50]
                dataset = Subset(dataset, rand_indx)

            # Split the dataset in training and validation
            training_data, validation_data = split_train_validation(dataset)

            train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
            dataloaders = {'train': train_dataloader, 'val': val_dataloader}

            # Detect if we have a GPU available
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(device)

            # Send the model to GPU
            model_ft = model_ft.to(device)

            params_to_update = model_ft.parameters()
            print("Params to learn:")
            if feature_extract:
                params_to_update = []
                for name, param in model_ft.named_parameters():
                    if param.requires_grad == True:
                        params_to_update.append(param)
                        print("\t", name)
            else:
                for name, param in model_ft.named_parameters():
                    if param.requires_grad == True:
                        print("\t", name)

            optimizer_ft = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9)

            criterion = nn.CrossEntropyLoss()

            # Train and evaluate
            model_ft, history = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=num_epochs)

            # Plot the train
            # for phase in ['train', 'val']:
            #     data = np.array(history[phase])
            #     x = np.arange(1, data.shape[0] + 1)
            #     plt.subplot(2, 1, 1)
            #     plt.plot(x, data[:, 0], label='{}'.format(phase), marker='.')
            #     plt.title('Loss')
            #     plt.xlabel('Epochs')
            #     plt.legend()
            #     plt.subplot(2, 1, 2)
            #     plt.plot(x, data[:, 1], label='{}'.format(phase), marker='.')
            #     plt.title('Accuracy')
            #     plt.xlabel('Epochs')
            # plt.legend()
            # plt.show()

            if SAVE:
                # Save the model
                f = open(f'{model_name}_{learning_rate}_{num_classes}_classes.pkl', 'wb')
                pickle.dump(model_ft, f)
                f.close()

                # Save the history
                f = open(f'history_{model_name}_{learning_rate}_{num_classes}_classes.pkl', 'wb')
                pickle.dump(history, f)
                f.close()
