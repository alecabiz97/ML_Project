import numpy as np
from matplotlib import pyplot as plt

def imshow(img):
    '''Show a tensor image'''
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()