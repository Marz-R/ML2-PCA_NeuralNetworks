from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import GridSearchCV


def calculate_mse(targets, predictions):
    """
    :param targets:
    :param predictions: Predictions obtained by using the model
    :return:
    """
    # TODO Calculate MSE using mean_squared_error from sklearn.metrics (alrady imported)
    mse = mean_squared_error(targets, predictions)
    return mse

def visualize_3D_data(x, y):
    """
    :param x: Datapoints - (x, y) coordinates. Shape: (n_samples, 2)
    :param y: Datapoints y. Shape: (n_samples, ). y = f(x1, x2)
    :return:
    """
    # TODO: 3D plot to illustrate data cloud (we want to see data points, not surface)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter3D(x[:,0], x[:,1], y, c='g', marker='o')

    ax.set_xlabel('x feature1')
    ax.set_ylabel('x feature2')
    ax.set_zlabel('y target')

    plt.show()
    
    pass

def solve_regression_task(features, targets):
    """
    :param features:
    :param targets:
    :return:
    """
    # TODO: MLPRegressor, choose the model yourself
    
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=33)
    
    nn = MLPRegressor(max_iter=7000)
    param_list = {"hidden_layer_sizes": [(1,),(20,),(50,)], "activation": ["logistic", "relu"], "solver": ["lbfgs", "adam"], "alpha": [0.00005,0.0005]}
    gridCV = GridSearchCV(estimator=nn, param_grid=param_list)

    ## Train the network
    gridCV.fit(X_train, y_train)

    # Calculate predictions
    y_pred_train = gridCV.predict(X_train)
    y_pred_test = gridCV.predict(X_test)
    print(f'Train MSE: {calculate_mse(y_train, y_pred_train):.4f}. Test MSE: {calculate_mse(y_test, y_pred_test):.4f}')

    print('Train R^2 Score : %.3f'%gridCV.best_estimator_.score(X_train, y_train))
    print('Test R^2 Score : %.3f'%gridCV.best_estimator_.score(X_test, y_test))
    print('Best R^2 Score Through Grid Search : %.3f'%gridCV.best_score_)
    print('Best Parameters : ',gridCV.best_params_)

