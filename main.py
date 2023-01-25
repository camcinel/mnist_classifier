import argparse
import network
import data
import image
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm


def pre_process_data(dataset, normalization_func, param_1=None, param_2=None):
    X, param_1, param_2 = normalization_func(dataset[0], param_1, param_2)
    X = data.append_bias(X)
    return (X, dataset[1]), param_1, param_2


def cross_validation(hyperparameters, int_1=None, int_2=None, softmax=True, grid_search=False):
    # load data
    X_train, y_train = data.load_data(os.path.join(os.path.dirname(__file__), 'data'))

    if not softmax:
        X_train, y_train = data.get_ints((X_train, y_train), int_1, int_2)

    # perform PCA
    pca = PCA(n_components=hyperparameters.p)
    pca.fit(X_train)
    X_train = pca.transform(X_train)

    if softmax:
        y_train = data.onehot_encode(y_train)

    # perform k-fold cross validation
    best_loss = np.inf
    count = 0
    loss_array_train = np.zeros((hyperparameters.k_folds, hyperparameters.epochs))
    loss_array_val = np.zeros((hyperparameters.k_folds, hyperparameters.epochs))
    acc_array_train = np.zeros((hyperparameters.k_folds, hyperparameters.epochs))
    acc_array_val = np.zeros((hyperparameters.k_folds, hyperparameters.epochs))
    for train, val in tqdm(data.generate_k_fold_set((X_train, y_train), k=hyperparameters.k_folds), desc=' k-folds',
                           position=0, total=hyperparameters.k_folds, leave=False):

        # generate instance of model
        if softmax:
            model = network.Network(hyperparameters, network.softmax, network.multiclass_cross_entropy, 10)
        else:
            model = network.Network(hyperparameters, network.sigmoid, network.binary_cross_entropy, 1)

        # data preprocessing for training
        train, param_1, param_2 = pre_process_data(train, hyperparameters.normalization)

        # data preprocessing for validation
        val, param_1, param_2 = pre_process_data(val, hyperparameters.normalization, param_1, param_2)

        # go through epochs
        for epoch in tqdm(range(hyperparameters.epochs), desc=' epochs', position=1, leave=False):
            train = data.shuffle(train)
            batch = 0
            loss_batch = np.zeros(np.ceil(len(train[0]) / hyperparameters.batch_size).astype(int))
            acc_batch = np.zeros(np.ceil(len(train[0]) / hyperparameters.batch_size).astype(int))

            # train on training set
            for mini_batch in data.generate_minibatches(train, batch_size=hyperparameters.batch_size):
                loss_batch[batch], acc_batch[batch] = model.train(mini_batch)
                batch += 1
            loss_array_train[count, epoch] = np.average(loss_batch)
            acc_array_train[count, epoch] = np.average(acc_batch)

            # test on validation set
            loss_array_val[count, epoch], acc_array_val[count, epoch] = model.test(val)

        # get best model
        if loss_array_val[count, -1] < best_loss:
            best_loss = loss_array_val[count, -1]
            best_model = model
            best_param_1, best_param_2 = param_1, param_2
            best_acc = acc_array_val[count, -1]
        count += 1

    if grid_search:
        return hyperparameters, best_acc
    else:
        if softmax:
            # plot the average loss per epoch for test and validation sets
            image.create_loss_plot(loss_array_train, loss_array_val, name='average_loss_softmax.png')

            # turn weights back into 784 dimensional array
            weights_nobias = np.transpose(best_model.weights[1:])
            if hyperparameters.normalization == data.z_score_normalize:
                weights_unnormal = (param_2 - param_1) * weights_nobias + param_1
            else:
                weights_unnormal = param_2 * weights_nobias + param_1
            true_weights = pca.inverse_transform(weights_unnormal)

            # create weight images
            image.export_image_plt(true_weights, 'weights.png')
        else:
            image.create_loss_plot(loss_array_train, loss_array_val, name='average_loss_' + str(int_1) + '_vs_'
                                                                          + str(int_2) + '.png')

        return best_model, pca, best_param_1, best_param_2


def test_model(model, pca, param_1, param_2, hyperparameters, int_1=None, int_2=None, softmax=True):
    X_test, y_test = data.load_data(os.path.join(os.path.dirname(__file__), 'data'), train=False)

    if not softmax:
        X_test, y_test = data.get_ints((X_test, y_test), int_1, int_2)

    # perform PCA
    X_test = pca.transform(X_test)

    (X_test, y_test), param_1, param_2 = pre_process_data((X_test, y_test), hyperparameters.normalization, param_1,
                                                          param_2)
    if softmax:
        y_test = data.onehot_encode(y_test)

    avg_loss, acc = model.test((X_test, y_test))
    print(f'Loss is {avg_loss}')
    print(f'Accuracy is {acc}')


def main(hyperparameters):
    if not hyperparameters.no_softmax:
        print('Softmax regression:')
        model, pca, param_1, param_2 = cross_validation(hyperparameters)
        test_model(model, pca, param_1, param_2, hyperparameters)
    if not hyperparameters.no_logistic:
        print('Logistic regression 2 vs 7:')
        model, pca, param_1, param_2 = cross_validation(hyperparameters, int_1=2, int_2=7, softmax=False)
        test_model(model, pca, param_1, param_2, hyperparameters, int_1=2, int_2=7, softmax=False)
        print('Logistic regression 5 vs 8:')
        model, pca, param_1, param_2 = cross_validation(hyperparameters, int_1=5, int_2=8, softmax=False)
        test_model(model, pca, param_1, param_2, hyperparameters, int_1=5, int_2=8, softmax=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSE251B PA1')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--z-score', dest='normalization', action='store_const',
                        default=data.min_max_normalize, const=data.z_score_normalize,
                        help='use z-score normalization on the dataset, default is min-max normalization')
    parser.add_argument('--k-folds', type=int, default=5,
                        help='number of folds for cross-validation')
    parser.add_argument('--p', type=int, default=100,
                        help='number of principal components')
    parser.add_argument('--no-softmax', action='store_false', help='do not perform softmax regression')
    parser.add_argument('--no-logistic', action='store_false', help='do not perform logistic regression')

    hyperparameters = parser.parse_args()
    os.makedirs('images', exist_ok=True)
    main(hyperparameters)
