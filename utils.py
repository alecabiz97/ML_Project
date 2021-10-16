import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import random_split
from secml.core.constants import inf
from icub_datasets import ICubWorld28
from torchvision import transforms
import pickle


def imshow(img):
    """
    Show a tensor image
    :param img: Tensor image
    :return: None
    """
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def split_train_validation(dataset, tr=0.8):
    """
    Split the data giving 80% to train and 20% to validation
    :param dataset: Dataset used
    :param tr: Percentage of training samples
    :return: training_data, validation_data
    """
    n_train_samples = int(tr * len(dataset))
    n_val_samples = len(dataset) - n_train_samples
    print(n_train_samples, n_val_samples)

    generator = torch.Generator().manual_seed(42)
    training_data, validation_data = random_split(dataset, [n_train_samples, n_val_samples], generator)
    return training_data, validation_data


def run_debug(clf, X, y, attack):
    """
    Visualizes the image of the input sample and the perturbed sample, along with the debugging
    information for the optimization.
    :param clf: instantiated secml classifier
    :param X: initial sample
    :param y: label of the sample X
    :param attack: instantiated attack from the secml library
    :return: None
    """
    dataset = ICubWorld28('ICubWorld28')
    dataset_labels = dataset.labels
    x0, y0 = X[0, :], y[0]
    y_pred_adv, _, adv_ds, _ = attack.run(x0, y0)
    from secml.figure import CFigure
    img_normal = x0.tondarray().reshape((3, 224, 224)).transpose(1, 2, 0)
    img_adv = adv_ds.X[0, :].tondarray().reshape((3, 224, 224)).transpose(1, 2, 0)

    diff_img = img_normal - img_adv
    diff_img -= diff_img.min()
    diff_img /= diff_img.max()

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225])])

    fig = CFigure(height=7, width=15)
    fig.subplot(1, 3, 1)
    fig.sp.imshow(data_transforms(img_normal).numpy().transpose(1, 2, 0))
    fig.sp.title('{0}'.format(dataset_labels[y0.item()]))
    fig.sp.xticks([])
    fig.sp.yticks([])

    fig.subplot(1, 3, 2)
    fig.sp.imshow(data_transforms(img_adv).numpy().transpose(1, 2, 0))
    fig.sp.title('{0}'.format(dataset_labels[y_pred_adv.item()]))
    fig.sp.xticks([])
    fig.sp.yticks([])

    fig.subplot(1, 3, 3)
    fig.sp.imshow(data_transforms(diff_img).numpy().transpose(1, 2, 0))
    fig.sp.title('Amplified perturbation')
    fig.sp.xticks([])
    fig.sp.yticks([])
    fig.tight_layout()

    # visualize the attack loss
    attack_loss = attack.objective_function(attack.x_seq)
    fig_loss = CFigure()
    fig_loss.sp.plot(attack_loss)
    fig_loss.sp.title("Attack loss")

    # visualize the perturbation size
    pert_size = (attack.x_seq - x0).norm_2d(axis=1, order=2)
    fig_pert_size = CFigure()
    fig_pert_size.sp.plot(pert_size)
    fig_pert_size.sp.title("Perturbation size (L-2)")

    # visualize the logits of all the classes
    fig_scores = CFigure()
    for cl_idx, cl in enumerate(dataset_labels):
        scores = clf.decision_function(attack.x_seq, cl_idx)
        fig_scores.sp.plot(scores, label=cl)
    fig_scores.sp.title("Scores")
    fig_scores.sp.legend()

    CFigure.show()


# Used one time to save the results in a better way
def compact_the_results(n_classes, models, learning_rates):
    """
    Compacts the results for better understanding
    :param n_classes: Number of classes used
    :param models: Models used
    :param learning_rates: Learning Rates used
    :return: None
    """
    for n in n_classes:
        train_results = {}
        for m in models:
            for lr in learning_rates:
                filename = f'history_{m}_{lr}_{n}_classes.pkl'
                f = open(f'results/{filename}', 'rb')
                x = pickle.load(f)
                f.close()
                train_results[f'{m}_{lr}'] = x

        f = open(f'results/train_results_{n}_classes.pkl', 'wb')
        pickle.dump(train_results, f)
        f.close()

    # f = open(f'results/train_results_7_classes.pkl', 'rb')
    # x = pickle.load(f)
    # f.close()


def make_subset(dataset, n_sample_for_class):
    """
    Creates a subset of the initial dataset
    :param dataset: Dataset used
    :param n_sample_for_class: Number of samples for each class
    :return: X, Tensor (n_sample_for_class * n_classes, 3, 224, 224)
    :return: Y, Tensor (n_sample_for_class * n_classes)
    """
    img_info = dataset.img_info
    labels = dataset.labels
    n_classes = len(labels)
    X = torch.zeros(n_sample_for_class * n_classes, 3, 224, 224)
    Y = torch.zeros(n_sample_for_class * n_classes, dtype=int)
    i = 0
    for label in labels:
        # print(f'{label}: {np.sum(img_info[:,1]==label)} sample')
        indices = np.argwhere(img_info[:, 1] == label)[0:n_sample_for_class].flatten()
        for idx in indices:
            x, y = dataset.__getitem__(idx)
            X[i, :, :, :] = x
            Y[i] = y
            i += 1
    return X, Y
