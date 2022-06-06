import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def reduce_dimension(features, n_components):
    """
    :param features: Data to reduce the dimensionality. Shape: (n_samples, n_features)
    :param n_components: Number of principal components
    :return: Data with reduced dimensionality. Shape: (n_samples, n_components)
    """

    pca = PCA(n_components = n_components, whiten = True, svd_solver = 'randomized')

    # TODO fit the model with features
    X_reduced = pca.fit_transform(features)
    # TODO apply the transformation on features

    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f'Explained variance: {explained_var}')
    return X_reduced

def train_nn(features, targets):
    """
    Train MLPClassifier with different number of neurons in one hidden layer.

    :param features:
    :param targets:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)

    n_hidden_neurons = [10, 100, 200]

    for i in range(len(n_hidden_neurons)):
        nn = MLPClassifier(hidden_layer_sizes = n_hidden_neurons[i], solver = 'adam', max_iter = 500, random_state = 1)
        print(f'Number of hidden neurons: {n_hidden_neurons[i]}')
        
        nn.fit(X_train, y_train)

        train_acc = nn.score(X_train, y_train)
        test_acc = nn.score(X_test, y_test)
        loss = nn.loss_

        print(f'Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')
        print(f'Loss: {loss:.4f}')
        print("--------------------")

def train_nn_with_regularization(features, targets):
    """
    Train MLPClassifier using regularization.

    :param features:
    :param targets:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)

    # Copy your code from train_nn, but experiment now with regularization (alpha, early_stopping).

    n_hidden_neurons = [10, 100, 200]

    for i in range(len(n_hidden_neurons)):
        nn = MLPClassifier(hidden_layer_sizes = n_hidden_neurons[i], solver = 'adam', max_iter = 500, random_state = 1, early_stopping = True)
        print(f'Number of hidden neurons: {n_hidden_neurons[i]}')
        
        nn.fit(X_train, y_train)

        train_acc = nn.score(X_train, y_train)
        test_acc = nn.score(X_test, y_test)
        loss = nn.loss_

        print(f'Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')
        print(f'Loss: {loss:.4f}')
        print("--------------------")

def train_nn_with_different_seeds(features, targets):
    """
    Train MLPClassifier using different seeds.
    Print confusion matrix and classification report.

    :param features:
    :param targets:
    :return:
    """
        #Print (mean +/- std) accuracy on the training and test set.

    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)
    seeds = [1, 10, 20, 50, 100]# TODO create a list of different seeds of your choice

    train_acc_arr = np.zeros(len(seeds))
    test_acc_arr = np.zeros(len(seeds))

    for i in range(len(seeds)):
        nn = MLPClassifier(hidden_layer_sizes = 200, solver = 'adam', max_iter = 500, random_state = seeds[i], early_stopping = True)
        nn.fit(X_train, y_train)
        print(f'Seed: {seeds[i]}')
        # TODO create an instance of MLPClassifier, check the perfomance for different seeds
        train_acc_arr[i] = nn.score(X_train, y_train)
        test_acc_arr[i] = nn.score(X_test, y_test)

        train_acc = nn.score(X_train, y_train) 
        test_acc =  nn.score(X_test, y_test)
        loss =  nn.loss_
        print(f'Train accuracy: {train_acc:.4f}. Test accuracy: {test_acc:.4f}')
        print(f'Loss: {loss:.4f}')
        print("--------------------")

    train_acc_mean = np.mean(train_acc_arr)
    train_acc_std = np.std(train_acc_arr)
    test_acc_mean = np.mean(test_acc_arr)
    test_acc_std = np.std(test_acc_arr)
    print(f'On the train set: {train_acc_mean:.4f} +/- {train_acc_std:.4f}')
    print(f'On the test set: {test_acc_mean:.4f} +/- {test_acc_std:.4f}')

    # TODO: print min and max accuracy as well
    max_train_acc = np.max(train_acc_arr)
    min_train_acc = np.min(train_acc_arr)
    max_test_acc = np.max(test_acc_arr)
    min_test_acc = np.min(test_acc_arr)
    print(f'Max Train Acc: {max_train_acc:.4f}  Min Train Acc: {min_train_acc:.4f}')
    print(f'Max Test Acc: {max_test_acc:.4f}  Min Test Acc: {min_test_acc:.4f}')

    nn = MLPClassifier(hidden_layer_sizes = 200, solver = 'adam', max_iter = 500, random_state = 50, early_stopping = True)
    nn.fit(X_train, y_train)

    plt.plot(nn.loss_curve_)
    plt.show()

    # TODO: Confusion matrix and classification report (for one classifier that performs well)
    print("Predicting on the test set")
    y_pred = nn.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred, labels=range(10)))


def perform_grid_search(features, targets):
    """
    Perform GridSearch using GridSearchCV.
    Create a dictionary of parameters, then a MLPClassifier (e.g., nn, set default values as specified in the HW2 sheet).
    Create an instance of GridSearchCV with parameters nn and dict.
    Print the best score and the best parameter set.

    :param features:
    :param targets:
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)
    parameters = {
        "alpha": [0.0, 0.001, 1.0, 10.0],
        "learning_rate_init": [0.001, 0.02],
        "solver": ['lbfgs', 'adam'],
        "hidden_layer_sizes": [(50,), (100,)]
    }

    nn = MLPClassifier(max_iter = 500, activation = 'logistic', random_state = 1, early_stopping = True)
    grid_search = GridSearchCV(nn, parameters, n_jobs = -1)

    grid_search.fit(X_train, y_train)

    best_score = grid_search.best_score_
    best_para = grid_search.best_params_
    print(f'Best Score: {best_score:.4f}')
    print(f'Best Parameters: {best_para}')

