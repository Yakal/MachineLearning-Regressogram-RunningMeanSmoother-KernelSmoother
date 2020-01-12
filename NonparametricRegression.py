__author__ = 'Furkan Yakal'
__email__ = 'fyakal16@ku.edu.tr'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

# Read the data points as float
data = np.array(pd.read_csv("hw04_data_set.csv", header=None))[1:].astype(float)
x_data = data[:, 0]
y_data = data[:, 1]

# training set is formed from first 150 data points
x_train = x_data[0:150]  # x coordinates of the training data
y_train = y_data[0:150]  # y coordinates of the training data

# test set is formed from remaining data points
x_test = x_data[150:]  # x coordinates of the test data
y_test = y_data[150:]  # y coordinates of the test data
N = len(y_test)  # length of the test set

colors = {"training": "blue", "test": "red", "line": "black"}  # colors of the points and line

# hyperparameters
h = 0.37
origin_parameter = 1.5


# root mean squared error
def rmse(y_head, y, length):
    return np.sqrt(np.sum((y - y_head) ** 2) / length)


def regressogram(x_values):
    bin_index = int((x_values - origin_parameter) / h)
    y_values_in_bin = y_train[
        np.where((x_train >= origin_parameter + bin_index * h) & (x_train < origin_parameter + (bin_index + 1) * h))]
    return sum(y_values_in_bin) / len(y_values_in_bin)


def running_mean_smoother(x_values):
    y_values_in_the_bin = y_train[np.where((x_train >= (x_values - h / 2)) & (x_train <= (x_values + h / 2)))]
    return np.sum(y_values_in_the_bin) / len(y_values_in_the_bin)


def kernel_smoother(x_values):
    u = (x_train - x_values) / h
    kernel_u = (1 / np.sqrt(2 * np.pi)) * np.exp(-(u ** 2) / 2)
    return np.dot(kernel_u, y_train) / np.sum(kernel_u)


def draw_plot(x_values, y_values, title):
    plot.subplot()
    plot.scatter(x_train, y_train, alpha=0.6, c=colors["training"], edgecolors='none', label="training")
    plot.scatter(x_test, y_test, alpha=0.6, c=colors["test"], edgecolors='none', label="test")
    plot.plot(x_values, y_values, c=colors["line"])
    plot.legend(loc=2)
    plot.xlabel('Eruption time (min)')
    plot.ylabel('Waiting time to next eruption (min)')
    plot.title(title)
    plot.show()


# performs each nonparametric method in the following order Regressogram, Running Mean Smoother, Kernel Smoother
def perform_nonparametric_regressions():
    #  the data interval for plotting the line properly
    data_interval_for_plotting = np.arange(min(x_train) - h / 4, max(x_train) + h / 4, 0.001)

    regressogram_vec = np.vectorize(regressogram)
    y_values = regressogram_vec(data_interval_for_plotting)
    draw_plot(data_interval_for_plotting, y_values, "Regressogram")
    rmse_regressogram = rmse(regressogram_vec(x_test), y_test, N)
    print("Running Mean Smoother => RMSE is {} when h is {}\n".format(rmse_regressogram, h))

    running_mean_smoother_vec = np.vectorize(running_mean_smoother)
    y_values = running_mean_smoother_vec(data_interval_for_plotting)
    draw_plot(data_interval_for_plotting, y_values, "Running Mean Smoother")
    rmse_running_mean_smoother = rmse(running_mean_smoother_vec(x_test), y_test, N)
    print("Running Mean Smoother => RMSE is {} when h is {}\n".format(rmse_running_mean_smoother, h))

    kernel_smoother_vec = np.vectorize(kernel_smoother)
    y_values = kernel_smoother_vec(data_interval_for_plotting)
    draw_plot(data_interval_for_plotting, y_values, "Kernel Smoother")
    rmse_kernel_smoother = rmse(kernel_smoother_vec(x_test), y_test, N)
    print("Kernel Smoother => RMSE is {} when h is {}\n".format(rmse_kernel_smoother, h))


if __name__ == "__main__":
    print("\n----------------------------------------------------------------------")
    perform_nonparametric_regressions()
    print("----------------------------------------------------------------------")
