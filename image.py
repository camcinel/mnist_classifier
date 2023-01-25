from PIL import Image
import os
import numpy as np
from matplotlib import pyplot as plt


def export_image(img_arr, name='test.png'):
    file_path = os.path.join(os.path.dirname(__file__), name)
    Image.fromarray(img_arr.reshape((28, 28)).astype(np.uint8), 'P').save(file_path)


def export_image_plt(img_arr, name='test.png'):
    file_path = os.path.join(os.path.dirname(__file__), 'images', name)
    plt.figure(3)
    for index, weight in enumerate(img_arr):
        plt.subplot(2, 5, index + 1)
        plt.title(index)
        plt.imshow(weight.reshape((28, 28)), interpolation='nearest', cmap='gray')
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
    plt.savefig(file_path)


def create_loss_plot(train_loss, val_loss, name='average_loss.png'):
    plt.figure()
    plt.title(name)
    plt.plot(np.average(train_loss, axis=0), label='train', linestyle='dashed')
    plt.plot(np.average(val_loss, axis=0), label='validation', linestyle='solid')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'images', name))
