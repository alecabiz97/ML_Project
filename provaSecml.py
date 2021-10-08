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


import pickle
from icub_datasets import ICubWorld28
from torchvision import transforms, models
from torch.utils.data import DataLoader

# this is for visualization
from secml.figure import CFigure

random_state = 999

metric = CMetricAccuracy()

f=open('trained_models/densenet161_0.001_28_classes.pkl','rb')
model = pickle.load(f)
f.close()
data_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
dataset = ICubWorld28('ICubWorld28', train=False, transform=data_transforms, sub_label=True)
labels = dataset.labels
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
X, Y = next(iter(dataloader))
X, Y = CArray(X.numpy()), CArray(Y.numpy())
clf = CClassifierPyTorch(model=model, input_shape=(3, 224, 224), pretrained=True, pretrained_classes=labels)

# y_pred = clf.predict(ts.X)
y_pred = clf.predict(X)

# Evaluate the accuracy of the classifier
# acc = metric.performance_score(y_true=ts.Y, y_pred=y_pred)
acc = metric.performance_score(y_true=Y, y_pred=y_pred)

print("Accuracy on test set: {:.2%}".format(acc))

X, Y = X.get_data(), Y.get_data()
x0, y0 = X[0], Y[0]  # Initial sample
steps = 100
epsilon = 0.02
y_target = None  # None if `error-generic`, the label of the target class for `error-specific`
lb = 0.0
ub = 1.0

pgd_attack = CFoolboxPGDL2(clf, y_target,
                           lb=lb, ub=ub,
                           epsilons=epsilon,
                           abs_stepsize=0.03,
                           steps=steps,
                           random_start=False)
y_pred, _, adv_ds_pgd, _ = pgd_attack.run(x0, y0)

print("Original x0 label: ", y0.item())
print("Adversarial example label (PGD-L2): ", y_pred.item())

# required for visualization in notebooks
#
# fig = CFigure(width=8, height=6, markersize=12)
# constraint = CConstraintL2(center=x0, radius=epsilon)  # visualize the constraint
#
# fig.sp.plot_fun(pgd_attack.objective_function, plot_levels=False,
#                 multipoint=True, n_grid_points=100)  # attack objective function
# fig.sp.plot_decision_regions(clf, plot_background=False,
#                              n_grid_points=200)  # decision boundaries
#
# # Construct an array with the original point and the adversarial example
# adv_path = x0.append(pgd_attack.x_seq, axis=0)
#
# fig.sp.plot_path(pgd_attack.x_seq)  # plots the optimization sequence
# fig.sp.plot_constraint(constraint)  # plots the constraint
#
# fig.sp.title(pgd_attack.class_type)
# fig.sp.grid(grid_on=False)
#
# fig.title(r"Error-generic evasion attack ($\varepsilon={:}$)".format(epsilon))
# fig.show()

# Perturbation levels to test
# e_vals = CArray.arange(start=0, step=0.05, stop=0.5)
#
# from secml.adv.seceval import CSecEval
# sec_eval = CSecEval(
#     attack=pgd_attack, param_name='epsilon', param_values=e_vals)
#
# # Run the security evaluation using the test set
# print("Running security evaluation...")
# sec_eval.run_sec_eval(ts)
#
# fig = CFigure(height=5, width=5)
#
# # Convenience function for plotting the Security Evaluation Curve
# fig.sp.plot_sec_eval(
#     sec_eval.sec_eval_data, marker='o', label='NN', show_average=True)
