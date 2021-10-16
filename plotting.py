import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from icub_datasets import ICubWorld28

num_classes = [28, 7]
for n in num_classes:
    # TEST DATA
    # Loading test results to show
    f = open(f'results/test_results_{n}_classes.pkl', 'rb')
    results = pickle.load(f)
    f.close()

    # Models used
    models = ['squeezenet1_1',
              'resnet152',
              'densenet161',
              'squeezenet1_0',
              'resnet18',
              'alexnet']
    learning_rates = ['0.01', '0.001', '0.0001']

    # Creation of lists for each model
    squeezenet1_1_acc = []
    resnet152_acc = []
    densenet161_acc = []
    squeezenet1_0_acc = []
    resnet18_acc = []
    alexnet_acc = []

    for m in models:
        for lr in learning_rates:
            if m == 'squeezenet1_1':
                squeezenet1_1_acc.append(round(results[f'{m}_{lr}'][1], 2))
            elif m == 'resnet152':
                resnet152_acc.append(round(results[f'{m}_{lr}'][1], 2))
            elif m == 'densenet161':
                densenet161_acc.append(round(results[f'{m}_{lr}'][1], 2))
            elif m == 'squeezenet1_0':
                squeezenet1_0_acc.append(round(results[f'{m}_{lr}'][1], 2))
            elif m == 'resnet18':
                resnet18_acc.append(round(results[f'{m}_{lr}'][1], 2))
            elif m == 'alexnet':
                alexnet_acc.append(round(results[f'{m}_{lr}'][1], 2))
    x = np.arange(len(learning_rates))  # the label locations
    width = 0.15  # the width of the bars

    plt.rc('axes', titlesize=20)  # fontsize of the axes title
    plt.rc('axes', labelsize=15)  # fontsize of the x and y labels
    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    fig.set_figheight(8)

    # rectangles shown while plotting the histogram
    rects1 = ax.bar(x - 2.5 * width, squeezenet1_1_acc, width, label='Squeezenet1_1')
    rects2 = ax.bar(x - 1.5 * width, resnet152_acc, width, label='Resnet152')
    rects3 = ax.bar(x - 0.5 * width, densenet161_acc, width, label='Densenet161')
    rects4 = ax.bar(x + 0.5 * width, squeezenet1_0_acc, width, label='Squeezenet1_0')
    rects5 = ax.bar(x + 1.5 * width, resnet18_acc, width, label='Resnet18')
    rects6 = ax.bar(x + 2.5 * width, alexnet_acc, width, label='Alexnet')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Learning Rates')
    ax.set_title(f'Accuracy by Architecture and Learning Rate with {n} Classes')
    ax.set_xticks(x)
    ax.set_xticklabels(learning_rates)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)
    ax.bar_label(rects5, padding=3)
    ax.bar_label(rects6, padding=3)

    plt.ylim([0, 100])

    fig.tight_layout()

    plt.show()

    # TRAINING DATA
    f = open(f'results/train_results_{n}_classes.pkl', 'rb')
    results = pickle.load(f)
    f.close()
    colors = ['#00ffff',
              '#000000',
              '#0000ff',
              '#ff00ff',
              '#00ff00',
              '#ff0000',
              '#778899',
              '#8a2be2',
              '#ff8c00',
              '#228b22',
              '#ffd700',
              '#fa8072']
    models = ['squeezenet1_1',
              'resnet152',
              'densenet161',
              'squeezenet1_0',
              'resnet18',
              'alexnet']
    learning_rates = [0.01, 0.001, 0.0001]
    for lr in learning_rates:
        plt.rc('axes', titlesize=15)  # fontsize of the axes title
        plt.rc('axes', labelsize=10)  # fontsize of the x and y labels
        fig1 = plt.figure(1)
        fig2 = plt.figure(2)
        i = 0
        for model in models:
            k = f'{model}_{lr}'
            d = results[k]
            training_data = np.array(d['train'])
            validation_data = np.array(d['val'])

            train_loss, train_acc = training_data[:, 0], training_data[:, 1]
            val_loss, val_acc = validation_data[:, 0], validation_data[:, 1]

            epochs = np.arange(len(train_loss))

            # ACCURACY
            plt.figure(1)
            fig1.set_figwidth(10)
            fig1.set_figheight(5)
            # plt.subplot(2, 1, 1)
            plt.plot(epochs, train_acc, label=f'train_{model}', linestyle='-', color=colors[i])
            plt.plot(epochs, val_acc, label=f'val_{model}', linestyle='--', color=colors[i + 1])
            # plt.suptitle(f'Learning Rate={lr} & Number of Classes={n}')
            plt.title(f'Learning Rate={lr} & Number of Classes={n}')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.ylim([0, 1])
            plt.legend(ncol=2)

            # LOSS
            plt.figure(2)
            fig2.set_figwidth(10)
            fig2.set_figheight(5)
            # plt.subplot(2, 1, 2)
            plt.plot(epochs, train_loss, label=f'train_{model}', linestyle='-', color=colors[i])
            plt.plot(epochs, val_loss, label=f'val_{model}', linestyle='--', color=colors[i + 1])
            plt.title(f'Learning Rate={lr} & Number of Classes={n}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend(ncol=2)
            plt.ylim([0, 0.045])

            i += 2
        plt.show()

# Confusion Matrix
for sub_label in [True, False]:
    num_classes = 28 if sub_label else 7
    f = open(f'results/test_results_{num_classes}_classes.pkl', 'rb')
    d = pickle.load(f)
    f.close()

    dataset = ICubWorld28('ICubWorld28', sub_label=sub_label)
    labels = dataset.labels
    learning_rates = ['0.01', '0.001', '0.0001']
    with pd.ExcelWriter(f'confusion_matrices_{num_classes}_classes.xlsx') as writer:
        for model_lr in d.keys():
            conf = d[model_lr][0]
            conf = np.around(conf.numpy(), 2)
            df = pd.DataFrame(conf, index=labels, columns=labels)
            df.to_excel(writer, sheet_name=model_lr, float_format="%.2f")
