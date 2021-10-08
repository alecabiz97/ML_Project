import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# TEST DATA
num_classes = [28, 7]
for n in num_classes:
    f = open(f'results/test_results_{n}_classes.pkl', 'rb')
    results = pickle.load(f)
    f.close()

    models = ['squeezenet1_1', 'resnet152', 'densenet161']
    learning_rates = ['0.01', '0.001', '0.0001']
    squeezenet1_1_acc = []
    resnet152_acc = []
    densenet161_acc = []
    for m in models:
        for lr in learning_rates:
            if m == 'squeezenet1_1':
                squeezenet1_1_acc.append(round(results[f'{m}_{lr}'][1], 2))
            elif m == 'resnet152':
                resnet152_acc.append(round(results[f'{m}_{lr}'][1], 2))
            elif m == 'densenet161':
                densenet161_acc.append(round(results[f'{m}_{lr}'][1], 2))

    x = np.arange(len(learning_rates))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, squeezenet1_1_acc, width, label='Squeezenet1_1')
    rects2 = ax.bar(x, resnet152_acc, width, label='Resnet152')
    rects3 = ax.bar(x + width, densenet161_acc, width, label='Densenet161')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Accuracy by architecture and learning rate {n} classes')
    ax.set_xticks(x)
    ax.set_xticklabels(learning_rates)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)

    fig.tight_layout()

    plt.show()

    # TRAINING DATA
    models = ['squeezenet1_1', 'resnet152', 'densenet161']
    learning_rates = [0.01, 0.001, 0.0001]
    for lr in learning_rates:
        fig = plt.figure()
        for model in models:
            s = f'results/history_{model}_{lr}_{n}_classes.pkl'
            f = open(s, 'rb')
            d = pickle.load(f)
            f.close()

            training_data = np.array(d['train'])
            validation_data = np.array(d['val'])

            train_loss, train_acc = training_data[:, 0], training_data[:, 1]
            val_loss, val_acc = validation_data[:, 0], validation_data[:, 1]

            epochs = np.arange(len(train_loss))

            fig.set_figwidth(10)
            fig.set_figheight(7)
            plt.subplot(2, 1, 1)
            plt.plot(epochs, train_acc, label=f'train_{model}', linestyle='-')
            plt.plot(epochs, val_acc, label=f'val_{model}', linestyle='--')
            plt.suptitle(f'Learning Rate={lr} & Number of Classes={n}')
            plt.ylabel('Accuracy')
            plt.ylim([0, 1])
            plt.legend(ncol=2)

            plt.subplot(2, 1, 2)
            plt.plot(epochs, train_loss, label=f'train_{model}', linestyle='-')
            plt.plot(epochs, val_loss, label=f'val_{model}', linestyle='--')
            # plt.title(f'Loss lr={lr}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend(ncol=2)
            plt.ylim([0, 0.045])

        plt.show()
