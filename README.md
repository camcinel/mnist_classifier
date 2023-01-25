# MNIST Classifier

---

## Description

This python module performs classification on the MNIST database done as part of PA1 for UCSD's CSE 251B.
Currently, it performs logistic regression on the pairs of integers 2 and 7 as well as 5 and 8.
It also performs a softmax regression on all 10 integer classes.
Both regressions are achieved through stochastic gradient descent.

For all three regressions, the module will create a chart of the average loss over each epoch,
located in the created director `/images`.
For the softmax regression, the weights will also be converted back into images and a chart of them will be created
in `/images`.

---

## Usage

To use this python module, first the data must be downloaded via the shell script
```bash
sh get_data.sh
```

Then the regressions can be run via
```commandline
python main.py
```

### Command Line arguments

This module supports the following command line arguments:

- `--batch-size`: changes the batch size for the mini-batch stochastic gradient descent.
The default is 1.
- `--epochs`: changes the number of epochs for training. The default is 100.
- `--learning-rate`: changes the learning rate of the gradient descent. The default is 0.001.
- `--z-score1`: changes the normalization function to z-score normalization. The default normalization is
min-max normalization.
- `--k-folds`: changes the number of folds in the k-folds crossvalidation procedure. The default is 5.
- `--p`: changes the number of principal components in the PCA decomposition. The default is 100.
- `--no-softmax`: does not perform softmax regression.
- `--no-logistic`: does not perform either logistic regression.

---

## Required Libraries

The following libraries are required to run the module:
- `scikit-learn` for PCA decomposition
- `matplotlib` to create the charts and weight images
- `numpy` to do the matrix algebra in the algorithm
- `tqdm` to create loading bars during the gradient descent
- `idx2numpy` to load the data into a numpy array

---

## File structure

- `get_data.sh`: shell script to download the MNIST training and test data.
- `data.py`: contains methods to load, normalize, and one hot encode data as well as create k-folds and mini-batches
- `network.py`: contains methods to define necessary functions for both regressions.
Also contains Network class to run the regression algorithms.
- `image.py`: contains methods to turn numpy arrays back into images as well as create loss charts.
- `main.py`: main python script to train and test the regressions
- `grid_search.py`: rudimentary script to perform a grid search on hyperparameters and find which one gives the best
accuracy. The hyperparameters to be grid searched over are contained in the dictionary `params`.