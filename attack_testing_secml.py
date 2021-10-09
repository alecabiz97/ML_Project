import torch
from torch import nn
from torch import optim
from secml.ml.classifiers import CClassifierPyTorch
from secml.ml.peval.metrics import CMetricAccuracy

# these are required for the dataset and preprocessing
from secml.data.loader import CDLRandomBlobs
from secml.data.splitter import CTrainTestSplit
from secml.ml.features import CNormalizerMinMax

# these are used for the adversarial attacks
from secml.optim.constraints import CConstraintL2
from secml.array import CArray
from secml.adv.attacks.evasion import CFoolboxPGDL2
from secml.adv.seceval import CSecEval

import pickle
from icub_datasets import ICubWorld28
from torchvision import transforms, models
from torch.utils.data import DataLoader

# this is for visualization
from secml.figure import CFigure
from utils import run_debug
from secml.data import CDataset
from time import time

n_samples=8000
sub_label=False
#Perturbation levels to test
e_vals = CArray.arange(start=0, step=50, stop=500)

for modelname in ['squeezenet1_0_0.01','resnet152_0.01','densenet161_0.001']:
    f=open(f'trained_models/{modelname}_7_classes.pkl','rb')
    model = pickle.load(f)
    f.close()

    data_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dataset = ICubWorld28('ICubWorld28', train=False, transform=data_transforms, sub_label=sub_label)
    labels = dataset.labels
    dataloader = DataLoader(dataset, batch_size=n_samples, shuffle=True)
    X, Y = next(iter(dataloader))
    print('Number of samples: ',len(Y))
    X, Y = CArray(X.numpy()), CArray(Y.numpy())
    ts=CDataset(X,Y)
    clf = CClassifierPyTorch(model=model, input_shape=(3, 224, 224), pretrained=True, pretrained_classes=labels)

    steps = 10
    epsilon = 300
    y_target = None  # None if `error-generic`, the label of the target class for `error-specific`
    lb = 0.0
    ub = 1.0

    pgd_attack = CFoolboxPGDL2(clf, y_target,
                               lb=lb, ub=ub,
                               epsilons=epsilon,
                               abs_stepsize=0.03,
                               steps=steps,
                               random_start=False)
    #run_debug(clf,X,Y,pgd_attack)


    sec_eval = CSecEval(
        attack=pgd_attack, param_name='epsilon', param_values=e_vals)

    # Run the security evaluation using the test set
    start=time()
    print("Running security evaluation...")
    sec_eval.run_sec_eval(ts)

    fig = CFigure(height=6, width=6)

    # Convenience function for plotting the Security Evaluation Curve
    fig.sp.plot_sec_eval(sec_eval.sec_eval_data, marker='o', label='NN', show_average=True)
    CFigure.show()
    end=time()
    print('Evaluation time: {} seconds'.format(round(end-start,2)))
    fig.savefig(f'eval_{modelname}.pdf',file_format='pdf')
