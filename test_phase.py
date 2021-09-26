import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from icub_datasets import ICubWorld28
from tqdm import tqdm
from torchmetrics import ConfusionMatrix
import numpy as np

def test(model, test_dataloader, criterion,n_classes):
    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    # For confusion matrix
    preds=[]
    targets=[]

    with torch.no_grad():
        for data in tqdm(test_dataloader):
            images, labels = data[0].to(device), data[1].to(device)

            targets.extend(labels.tolist())
            outputs = model(images)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            preds.extend(predicted.tolist())
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    #test_loss = running_loss / len(test_dataloader)
    accuracy = 100. * correct / total

    conf=ConfusionMatrix(num_classes=n_classes,normalize='true')
    confusion_matrix=conf(torch.tensor(preds),torch.tensor(targets))

    print('\nTest Accuracy: %.3f' % (accuracy))

    return confusion_matrix, accuracy


if __name__ == "__main__":

    num_classes = 7
    root = "iCubWorld28"
    results={}
    for model_name in ['resnet', 'alexnet', 'squeezenet']:
        for learning_rate in [0.01, 0.001, 0.0001]:
            batch_size = 64  # Batch size for training (change depending on how much memory you have)

            if model_name == 'inception':
                input_size = 299
            else:
                input_size = 224

            if torch.cuda.is_available():
                print("CUDA is available! Training on GPU...")
                device = "cuda"
            else:
                print("CUDA is not available. Training on CPU...")

            # Load the model
            f = open(f'{model_name}_{learning_rate}.pkl', 'rb')
            model = pickle.load(f)
            f.close()

            model.to(device)

            data_transforms = transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

            test_data = ICubWorld28(root, train=False, transform=data_transforms)
            # rand_indx = np.random.permutation(np.arange(len(test_data)))[0:20]
            # test_data = Subset(test_data, rand_indx)

            test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
            criterion = nn.CrossEntropyLoss()

            confusion_matrix, test_accuracy = test(model, test_dataloader, criterion,num_classes)
            filename=f'{model_name}_{learning_rate}'
            results[filename]=(confusion_matrix, test_accuracy)

    # Save the results
    f=open('results.pkl','wb')
    pickle.dump(results,f)
    f.close()