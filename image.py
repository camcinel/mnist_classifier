from PIL import Image
import os
import numpy as np
from matplotlib import pyplot as plt
import data
from sklearn.decomposition import PCA


def export_image(img_arr, name='test.png'):
    file_path = os.path.join(os.path.dirname(__file__), name)
    Image.fromarray(img_arr.reshape((28, 28)).astype(np.uint8), 'P').save(file_path)


def export_image_plt(img_arr, name='test.png'):
    """

    Parameters
    ----------
    img_arr (np.array) : numpy array of size (10, 784)
    name (str) : file name of save chart

    pyplot saves figure at /images/name

    """
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
    """
    Creates loss plot over training epochs

    Parameters
    ----------
    train_loss (np.array) : array of average losses on the training set for each epoch
    val_loss (np.array) : array of average losses on the validation set for each epoch
    name (str) : file name of saved chart

    pyplot saves figure at /images/name

    """
    plt.figure()
    plt.title(name)
    plt.plot(np.average(train_loss, axis=0), label='train', linestyle='dashed')
    plt.plot(np.average(val_loss, axis=0), label='validation', linestyle='solid')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'images', name))
    
def make_PCA_figure_1(trainBool,img_idx,pvals,errBool):
    """
    Inputs:
    trainBool (Boolean): use the training data (True) or the test data (False)
    img_idx (int): the random index of the image in the dataset to use
    pvals (list): a list of the 7 p values to apply to the image
    errBool (Boolean): True if you want the absolute error of reconstructed image with original image
    
    pyplot shows a 4x2 image of the 7 reconstructed images plus the original one.
    """
    fig, ax = plt.subplots(4,2)
    fig.set_size_inches(5,7)
    fig.suptitle('Original and reconstructed images')
    X_test, y_test = data.load_data(os.path.join(os.path.dirname(__file__), 'data'), train=trainBool)
    pltcnt=0
    for p in pvals:
        pca = PCA(n_components=p)#hyperparameters.p)
        pca.fit(X_test)
        transformedTestData=pca.transform(X_test)
        retransformedData=pca.inverse_transform(transformedTestData)
        ax[pltcnt//2][pltcnt%2].set_title('p='+str(p))
        if errBool:
            ax[pltcnt//2][pltcnt%2].pcolor(np.abs(retransformedData[img_idx].reshape(28,28)-X_test[img_idx].reshape(28,28)),cmap='gray',vmin=0,vmax=200)
        else:
            ax[pltcnt//2][pltcnt%2].pcolor(retransformedData[img_idx].reshape(28,28),cmap='gray',vmin=0,vmax=200)
        ax[pltcnt//2][pltcnt%2].set_axis_off()
        pltcnt+=1
    ax[3][1].pcolor(X_test[img_idx].reshape(28,28),cmap='gray',vmin=0,vmax=200)
    ax[3][1].set_title("original image")
    ax[3][1].set_axis_off()
    #plt.savefig('BLthickness_'+MODS[i][:-1]+'.png',dpi=300)
    plt.show()
    plt.clf()
    plt.close()
    
    
def make_PCA_figure_2(trainBool,errBool):
    fig, ax = plt.subplots(10,11)
    fig.set_size_inches(7,7)
    X, y = data.load_data(os.path.join(os.path.dirname(__file__), 'data'), train=trainBool)
    #print(np.argwhere(y==2).flatten())
    idx_list=[np.random.choice(np.argwhere(y==ii).flatten()) for ii in range(10)]
    ax[0][10].set_title("original")
    for p in np.arange(10): # integer class
        pca = PCA(n_components=p)#hyperparameters.p)
        pca.fit(X)
        transformedTestData=pca.transform(X)
        retransformedData=pca.inverse_transform(transformedTestData)
        print(np.shape(retransformedData))
        ax[0][p].set_title('p='+str(p))
        for i in np.arange(10): # p value
            idx=idx_list[i]
            ax[i][p].pcolor(retransformedData[idx].reshape(28,28),cmap='gray',vmin=0,vmax=200)
            ax[i][p].set_axis_off()
            ax[i][p].set_ylim(ax[i][p].get_ylim()[::-1])
            ax[i][10].pcolor(X[idx].reshape(28,28),cmap='gray',vmin=0,vmax=200)
            ax[i][10].set_axis_off()
            ax[i][10].set_ylim([28,0])
    plt.show()
    plt.clf()
    plt.close()
    
    
def plot_PCA_10(trainBool, pval):
    """
    trainBool: use the training data or the testing data?
    pval: the max number of PCA components to plot for the 2-7, 5-8, and all-class datasets
    
    This function produces a pvalx3 figure of the principle components of the three groups of datasets
    """
    fig, ax = plt.subplots(pval,3)
    fig.set_size_inches(4,8)
    fig.suptitle('PCA components from data classes:')
    X, y = data.load_data(os.path.join(os.path.dirname(__file__), 'data'), train=trainBool)
    class_2_7_data = data.get_ints((X,y),2,7) # pick out the 2s and 7s 
    class_5_8_data = data.get_ints((X,y),5,8) # pick out the 5s and 8s
    
    # All examples of the dataset
    pca_all = PCA(n_components=pval)#hyperparameters.p)
    pca_all.fit(X) # fit to the whole dataset
    components_all=pca_all.components_
    print(np.min(components_all[0]))
    print(np.max(components_all[0]))
    ax[0][0].set_title('all')
    for pc in range(pval): # plot each of the pval PCA components in 28x28 image
        ax[pc][0].pcolor(components_all[pc].reshape(28,28),cmap='gray',vmin=-0.2,vmax=0.2)
        ax[pc][0].set_axis_off()
        ax[pc][0].set_ylim(ax[pc][0].get_ylim()[::-1])

    # 2s and 7s
    pca_2_7 = PCA(n_components=pval)#hyperparameters.p)
    pca_2_7.fit(class_2_7_data[0])
    components_2_7=pca_2_7.components_
    ax[0][1].set_title('2s & 7s')
    for pc in range(pval): # plot each of the pval PCA components in 28x28 image
        ax[pc][1].pcolor(components_2_7[pc].reshape(28,28),cmap='gray',vmin=-0.2,vmax=0.2)
        ax[pc][1].set_axis_off()
        ax[pc][1].set_ylim(ax[pc][1].get_ylim()[::-1])

    # 5s and 8s
    pca_5_8 = PCA(n_components=pval)#hyperparameters.p)
    pca_5_8.fit(class_5_8_data[0])
    components_5_8=pca_5_8.components_
    ax[0][2].set_title('5s & 8s')
    for pc in range(pval): # plot each of the pval PCA components in 28x28 image
        ax[pc][2].pcolor(components_5_8[pc].reshape(28,28),cmap='gray',vmin=-0.2,vmax=0.2)
        ax[pc][2].set_axis_off()
        ax[pc][2].set_ylim(ax[pc][2].get_ylim()[::-1])

    plt.show()
    plt.clf()
    plt.close()
