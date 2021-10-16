import torch
from torch import nn
from torch import optim
from secml.ml.classifiers import CClassifierPyTorch
from secml.ml.peval.metrics import CMetricAccuracy

# these are used for the adversarial attacks
from secml.array import CArray
from secml.adv.attacks.evasion import CFoolboxPGDL2
from secml.adv.seceval import CSecEval

import pickle
from icub_datasets import ICubWorld28
from torchvision import transforms, models
from torch.utils.data import DataLoader

# this is for visualization
from secml.figure import CFigure
from utils import run_debug, make_subset
from secml.data import CDataset
from time import time
from secml.ml.features import CNormalizerMeanStd

# Hyperparameters
n_samples_for_class = 100
sub_label = False
steps = 50
y_target = None  # None if `error-generic`, the label of the target class for `error-specific`
lb = 0.0  # Lower bound
ub = 1.0  # Upper bound
abs_stepsize = 0.03

# Perturbation levels to test
e_vals = CArray.arange(start=0, step=0.05, stop=0.35)

fig = CFigure(height=6, width=6)

for modelname, lr in zip(['squeezenet1_0', 'resnet152', 'densenet161'], ['0.01', '0.01', '0.001']):
    # Load the chosen models
    f = open(f'trained_models/{modelname}_{lr}_7_classes.pkl', 'rb')
    model = pickle.load(f)
    f.close()

    # Data transformation
    data_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
    # Data normalization
    normalizer = CNormalizerMeanStd(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    dataset = ICubWorld28('ICubWorld28', train=False, transform=data_transforms, sub_label=sub_label)
    labels = dataset.labels
    X, Y = make_subset(dataset, n_samples_for_class)
    print('Number of samples: ', len(Y))
    X, Y = CArray(X.numpy()), CArray(Y.numpy())
    ts = CDataset(X, Y)
    clf = CClassifierPyTorch(model=model,
                             input_shape=(3, 224, 224),
                             preprocess=normalizer,
                             pretrained=True,
                             pretrained_classes=labels)

    # Chosen type of attack
    pgd_attack = CFoolboxPGDL2(clf, y_target,
                               lb=lb, ub=ub,
                               abs_stepsize=abs_stepsize,
                               steps=steps,
                               random_start=False)

    # run_debug(clf,X,Y,pgd_attack)

    sec_eval = CSecEval(attack=pgd_attack, param_name='epsilon', param_values=e_vals)

    # Run the security evaluation using the test set
    start = time()
    print("Running security evaluation...")
    sec_eval.run_sec_eval(ts)

    # Convenience function for plotting the Security Evaluation Curve
    fig.sp.plot_sec_eval(sec_eval.sec_eval_data, marker='o', label=f'{modelname}',
                         show_average=True, percentage=True)
    end = time()
    print('Evaluation time: {} minutes'.format(round((end - start) / 60, 2)))

# Plotting the results of the Attack
fig.sp.ylim(0, 100)
fig.sp.ylabel("Accuracy (%)")
CFigure.show()

# Saving the Attack Results on a pdf file
fig.savefig(f'Security Evaluation Curve.pdf', file_format='pdf')
