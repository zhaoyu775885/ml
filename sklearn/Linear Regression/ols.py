#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:36:10 2017

@author: zhaoyu
"""

print(__doc__)


# Code source: Jaques Grobler
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Use only one feature
diabetes_X = diabetes.data[:, 2:3]

# Define the training proportion
training_proportion = 0.9
n_samples = diabetes.data.shape[0]
n_train = round(n_samples * training_proportion)

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:n_train]
diabetes_X_test = diabetes_X[n_train:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:n_train, np.newaxis]
diabetes_y_test = diabetes.target[n_train:, np.newaxis]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# The TSS
TSS = np.sum( (diabetes_y_test - np.mean(diabetes_y_test)) ** 2 )
print('Total sum of squares: {0:.2f}'.format( TSS ))
# The RSS
RSS = np.sum( (regr.predict(diabetes_X_test) - diabetes_y_test)**2 )
print('Residual sum of squares: {0:.2f}'.format( RSS ))
# The R^2
print('R^2: {0:.6f}'.format((TSS-RSS)/TSS))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.6f\n' % regr.score(diabetes_X_test, diabetes_y_test))

# The mean squared error
print('Mean squared error: %.2f'
      % np.mean((regr.predict(diabetes_X_train) - diabetes_y_train) ** 2))
# The TSS
TSS = np.sum( (diabetes_y_train - np.mean(diabetes_y_train)) ** 2 )
print('Total sum of squares: {0:.2f}'.format( TSS ))
# The RSS
RSS = np.sum( (regr.predict(diabetes_X_train) - diabetes_y_train)**2 )
print('Residual sum of squares: {0:.2f}'.format( RSS ))
# The R^2
print('R^2: {0:.6f}'.format((TSS-RSS)/TSS))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.6f' % regr.score(diabetes_X_train, diabetes_y_train))

x = diabetes_X_train - np.mean(diabetes_X_train)
y = diabetes_y_train - np.mean(diabetes_y_train)
correlation_x_y = np.sum(x*y)/np.sqrt((np.sum(x**2)*np.sum(y**2)))
print('Correlation: {0:.6f}'.format(correlation_x_y**2))

'''
# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
         linewidth=3)

#plt.xticks(())
#plt.yticks(())

plt.show()
'''