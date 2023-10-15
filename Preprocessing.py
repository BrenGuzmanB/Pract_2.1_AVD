"""
Created on Sat Oct 14 00:10:10 2023

@author: Bren Guzmán, Brenda García, María José Merino
"""
# Librerías
import numpy as np
import math
import pandas as pd

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

def filter_box_cox(df, column_name, range=[-3, 3]):

    # Calcula la mediana y la desviación estándar de la columna
    median = df[column_name].median()
    std = df[column_name].std()

    # Calcula los valores z
    z_scores = (df[column_name] - median) / std

    # Elimina los valores z fuera del rango especificado
    filtered_df = df[np.abs(z_scores) <= range[1]]

    return filtered_df

def rule_of_two_sigmas(df, column_name):
    # Calcular la media y la desviación estándar de la columna
    column = df[column_name]
    mean = column.mean()
    std_dev = column.std()

    # Definir el umbral superior e inferior para valores atípicos
    upper_threshold = mean + 2 * std_dev
    lower_threshold = mean - 2 * std_dev

    # Identificar valores atípicos
    outliers = df[(column > upper_threshold) | (column < lower_threshold)]

    return outliers


























def one_hot_encoding(dataframe, column_name):
    
    column = dataframe[column_name]
    unique_values = column.unique()
    
    # Diccionario para almacenar las nuevas columnas
    one_hot_dict = {}
    
    # Recorremos los valores únicos y generamos las nuevas columnas
    for value in unique_values:
        one_hot_dict[f'{column_name}_{value}'] = (column == value).astype(int)
    
    # DataFrame con las columnas 
    one_hot_df = pd.DataFrame(one_hot_dict)
    
    # Concatenar el nuevo DataFrame con el original
    result_df = pd.concat([dataframe, one_hot_df], axis=1)
    
    # Eliminar la columna original
    result_df.drop(column_name, axis=1, inplace=True)
    
    return result_df
