"""
Created on Sat Oct 14 00:10:10 2023

@author: Bren Guzmán, Brenda García, María José Merino
"""
# Librerías
import numpy as np
import math

def box_cox_transform(data, lmbda):
    if lmbda == 0:
        # Si lambda es 0, aplicamos la transformación logarítmica
        return [math.log(x) for x in data]
    else:
        # Si lambda no es 0, aplicamos la transformación de Box-Cox
        return [((x ** lmbda) - 1) / lmbda if x > 0 else 0 for x in data]
    
def log_likelihood(data, lmbda):
    transformed_data = box_cox_transform(data, lmbda)
    n = len(data)
    log_lik = -(n / 2) * np.log(np.var(transformed_data)) + (lmbda - 1) * sum(np.log(data))
    return log_lik

def find_optimal_lambda(data):
    best_lambda = None
    best_log_likelihood = float("-inf")

    for lmbda in np.arange(0.1, 2.1, 0.1):  # Probamos valores de lambda en un rango
        lik = log_likelihood(data, lmbda)

        if lik > best_log_likelihood:
            best_log_likelihood = lik
            best_lambda = lmbda

    return best_lambda
