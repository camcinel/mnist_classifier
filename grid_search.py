import main
import data
import argparse
import itertools

param_grid = {
    'learning-rate': [.1, .01, .001, .0001],
    'normalization': [data.min_max_normalize, data.z_score_normalize],
    'batch-size': [10 ,100, 1000],
    'p': [10, 30, 100, 300]
}

def create_grid(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


parameters = argparse.Namespace()
parameters.epochs = 100
parameters.k_folds = 5
parameters.no_softmax = False
parameters.no_logistic = False

best_acc = 0
for params in create_grid(**param_grid):
    parameters.learning_rate = params['learning-rate']
    parameters.normalization = params['normalization']
    parameters.batch_size = params['batch-size']
    parameters.p = params['p']
    hyperparameters, acc = main.cross_validation(parameters, int_1=2, int_2=7, softmax=parameters.softmax, grid_search=True)
    if acc > best_acc:
        best_acc = acc
        best_hyperparameters = hyperparameters

print(f'Best accuracy is {best_acc}')
print('Best hyperparameters are: ')
print(best_hyperparameters)