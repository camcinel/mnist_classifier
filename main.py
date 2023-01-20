import argparse
import network
import data
import image
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

def main(hyperparameters):
    # load data
    X_train, y_train = data.load_data(os.path.join(os.path.dirname(__file__), 'data'))

    # normalize data
    X_train, mean, std = hyperparameters.normalization(X_train)

    # perform PCA
    pca = PCA(n_components = hyperparameters.p)
    pca.fit(X_train)
    X_train = pca.transform(X_train)

    # shuffle dataset
    X_train, y_train = data.shuffle((X_train, y_train))

    # append bias
    X_train = data.append_bias(X_train)

    # onehot encode
    y_train = data.onehot_encode(y_train)



    # perform k-fold cross validation
    best_loss = np.inf
    count = 0
    loss_array_train = np.zeros((hyperparameters.k_folds, hyperparameters.epochs))
    loss_array_val = np.zeros((hyperparameters.k_folds, hyperparameters.epochs))
    acc_array_train = np.zeros((hyperparameters.k_folds, hyperparameters.epochs))
    acc_array_val = np.zeros((hyperparameters.k_folds, hyperparameters.epochs))
    for train, val in tqdm(data.generate_k_fold_set((X_train, y_train), k = hyperparameters.k_folds), desc=' k-folds',
                           position=0):
        # generate instance of model
        softmax_reg = network.Network(hyperparameters, network.softmax, network.multiclass_cross_entropy, 10)

        # go through epochs
        for epoch in tqdm(range(hyperparameters.epochs), desc=' epochs', position=1, leave=False):
            train = data.shuffle(train)
            batch = 0
            loss_batch = np.zeros(np.ceil(len(train[0]) / hyperparameters.batch_size).astype(int))
            acc_batch = np.zeros(np.ceil(len(train[0]) / hyperparameters.batch_size).astype(int))

            # train on training set
            for mini_batch in data.generate_minibatches(train, batch_size = hyperparameters.batch_size):
                loss_batch[batch], acc_batch[batch] = softmax_reg.train(mini_batch)
                batch += 1
            loss_array_train[count, epoch] = np.average(loss_batch)
            acc_array_train[count, epoch] = np.average(acc_batch)

            # test on validation set
            loss_array_val[count, epoch], acc_array_val[count, epoch] = softmax_reg.test(val)

        # get best model
        if loss_array_val[count, -1] < best_loss:
            best_loss = loss_array_val[count, -1]
            best_model = softmax_reg
        count += 1

    # plot the average loss per epoch for test and validation sets
    plt.figure(1)
    plt.title('average loss')
    plt.plot(np.average(loss_array_train, axis=0), label='train', linestyle='dashed')
    plt.plot(np.average(loss_array_val, axis=0), label='validation', linestyle='solid')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'images', 'k_fold_loss_average.png'))

    true_weights = pca.inverse_transform(np.transpose(best_model.weights[1:]))
    # true_weights, w_min, w_max = data.min_max_normalize(true_weights)
    # true_weights = (_max - _min) * true_weights + _min

    image.export_image_plt(true_weights, 'weights.png')

    # matplotlib plotting
    '''
    plt.figure(1)
    plt.legend()
    plt.savefig('loss.png')
    plt.figure(2)
    plt.legend()
    plt.savefig('accuracy.png')
    '''
    # load data
    X_test, y_test = data.load_data(os.path.join(os.path.dirname(__file__), 'data'), train = False)

    # normalize data
    X_test, mean, std = hyperparameters.normalization(X_test, mean, std)

    # perform PCA
    X_test = pca.transform(X_test)

    # append bias
    X_test = data.append_bias(X_test)

    # onehot encode
    y_test = data.onehot_encode(y_test)

    avg_loss, acc = best_model.test((X_test, y_test))
    print('Loss is ' + str(avg_loss))
    print('Accuracy is ' + str(acc))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'CSE251B PA1')
    parser.add_argument('--batch-size', type = int, default = 1,
            help = 'input batch size for training (default: 1)')
    parser.add_argument('--epochs', type = int, default = 100,
            help = 'number of epochs to train (default: 100)')
    parser.add_argument('--learning-rate', type = float, default = 0.001,
            help = 'learning rate (default: 0.001)')
    parser.add_argument('--z-score', dest = 'normalization', action='store_const', 
            default = data.min_max_normalize, const = data.z_score_normalize,
            help = 'use z-score normalization on the dataset, default is min-max normalization')
    parser.add_argument('--k-folds', type = int, default = 5,
            help = 'number of folds for cross-validation')
    parser.add_argument('--p', type = int, default = 100,
            help = 'number of principal components')

    hyperparameters = parser.parse_args()
    os.makedirs('images', exist_ok=True)
    main(hyperparameters)
