import idx2numpy
import numpy as np
import os
import image
from sklearn.decomposition import PCA
import pickle


def load_data(data_directory, train=True):
    if train:
        images = idx2numpy.convert_from_file(os.path.join(data_directory, 'train_images'))
        labels = idx2numpy.convert_from_file(os.path.join(data_directory, 'train_labels'))
    else:
        images = idx2numpy.convert_from_file(os.path.join(data_directory, 'test_images'))
        labels = idx2numpy.convert_from_file(os.path.join(data_directory, 'test_labels'))

    vdim = images.shape[1] * images.shape[2]
    vectors = np.empty([images.shape[0], vdim])
    for imnum in range(images.shape[0]):
        imvec = images[imnum, :, :].reshape(vdim, 1).squeeze()
        vectors[imnum, :] = imvec

    return vectors, labels


def z_score_normalize(X, u=None, sd=None):
    """
    Performs z-score normalization on X. 

    f(x) = (x - μ) / σ
        where 
            μ = mean of x
            σ = standard deviation of x

    Parameters
    ----------
    X : np.array
        The data to z-score normalize
    u (optional) : np.array
        The mean to use when normalizing
    sd (optional) : np.array
        The standard deviation to use when normalizing

    Returns
    -------
        Tuple:
            Transformed dataset with mean 0 and stdev 1
            Computed statistics (mean and stdev) for the dataset to undo z-scoring.
    """
    if u is None:
        u = X.mean(axis=0)
    if sd is None:
        sd = X.std(axis=0)

    sd[sd == 0] = 1

    X_new = (X - u) / sd

    return X_new, u, sd


def min_max_normalize(X, _min=None, _max=None):
    """
    Performs min-max normalization on X. 

    f(x) = (x - min(x)) / (max(x) - min(x))

    Parameters
    ----------
    X : np.array
        The data to min-max normalize
    _min (optional) : np.array
        The min to use when normalizing
    _max (optional) : np.array
        The max to use when normalizing

    Returns
    -------
        Tuple:
            Transformed dataset with all values in [0,1]
            Computed statistics (min and max) for the dataset to undo min-max normalization.
    """
    if _min is None:
        _min = X.min(axis=0)
    if _max is None:
        _max = X.max(axis=0)

    diff = _max - _min
    diff[diff == 0] = 1

    X_new = (X - _min) / diff

    return X_new, _min, _max


def onehot_encode(y):
    """
    Performs one-hot encoding on y.

    Ideas:
        NumPy's `eye` function

    Parameters
    ----------
    y : np.array
        1d array (length n) of targets (k)

    Returns
    -------
        2d array (shape n*k) with each row corresponding to a one-hot encoded version of the original value.
    """
    return np.eye(y.max() + 1)[y]


def onehot_decode(y):
    """
    Performs one-hot decoding on y.

    Ideas:
        NumPy's `argmax` function 

    Parameters
    ----------
    y : np.array
        2d array (shape n*k) with each row corresponding to a one-hot encoded version of the original value.

    Returns
    -------
        1d array (length n) of targets (k)
    """
    return np.argmax(y, axis=1)


def shuffle(dataset):
    """
    Shuffle dataset.

    Make sure that corresponding images and labels are kept together. 
    Ideas: 
        NumPy array indexing 
            https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing

    Parameters
    ----------
    dataset
        Tuple containing
            Images (X)
            Labels (y)

    Returns
    -------
        Tuple containing
            Images (X)
            Labels (y)
    """
    sigma = np.random.permutation(dataset[0].shape[0])

    X_shuffled = dataset[0][sigma]
    y_shuffled = dataset[1][sigma]

    return X_shuffled, y_shuffled


def append_bias(X):
    """
    Append bias term for dataset.

    Parameters
    ----------
    X
        2d numpy array with shape (N,d)

    Returns
    -------
        2d numpy array with shape ((N+1),d)
    """
    return np.insert(X, 0, 1, axis=1)


def generate_minibatches(dataset, batch_size=64):
    X, y = dataset
    l_idx, r_idx = 0, batch_size
    while r_idx < len(X):
        yield X[l_idx:r_idx], y[l_idx:r_idx]
        l_idx, r_idx = r_idx, r_idx + batch_size

    yield X[l_idx:], y[l_idx:]


def generate_k_fold_set(dataset, k=5):
    X, y = dataset
    if k == 1:
        yield (X, y), (X[len(X):], y[len(y):])
        return

    order = np.random.permutation(len(X))

    fold_width = len(X) // k

    l_idx, r_idx = 0, fold_width

    for i in range(k):
        train = np.concatenate([X[order[:l_idx]], X[order[r_idx:]]]), np.concatenate(
            [y[order[:l_idx]], y[order[r_idx:]]])
        validation = X[order[l_idx:r_idx]], y[order[l_idx:r_idx]]
        yield train, validation
        l_idx, r_idx = r_idx, r_idx + fold_width


def get_ints(dataset, int_1, int_2):
    X, y = dataset

    if int_1 == int_2:
        raise Exception('Integer values must not be equal')

    full_data = np.append(X, y.reshape(-1,1), axis=1)
    specific_data = full_data[np.where((full_data[:, -1] == int_1) + (full_data[:, -1] == int_2))]

    X_new, y_new = specific_data[:, :-1], specific_data[:, -1]

    mask = y_new == int_1
    y_new[mask] = 1
    y_new[~mask] = 0
    
    return X_new, y_new.flatten()



if __name__ == '__main__':
    X_train, y_train = load_data(os.path.join(os.path.dirname(__file__), 'data'))

    X_train, mean, std = z_score_normalize(X_train)

    pca = PCA(n_components=30)
    pca.fit(X_train)
    X_pca = pca.transform(X_train)
    X_inv = pca.inverse_transform(X_pca)

    for number in range(10):
        image.export_image(X_train[number].reshape((28, 28)), 'original' + str(number) + '.tiff')
        image.export_image(X_inv[number].reshape((28, 28)), 'pca' + str(number) + '.tiff')
