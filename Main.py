
"""
Created on Sat Oct 14 00:12:30 2023

@author: Bren Guzmán, Brenda García, María José Merino
"""

#%% LIBRERÍAS
import pandas as pd
from Preprocessing import find_optimal_lambda, box_cox_transform, filter_box_cox
import seaborn as sns
import matplotlib.pyplot as plt
from Preprocessing import rule_of_two_sigmas as two_sigma
from Preprocessing import min_max_normalization as min_max, impute_median
from Preprocessing import one_hot_encoding
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


#%% CARGAR ARCHIVO

df = pd.read_csv('salariesSample.csv')


#%% EXPLORACIÓN
#%%%% Tipos de datos, conteo de valores nulos, etc
print("\n\nDescribe: \n",df.describe()) #estadísticos básicos
print("\n\n NaN Values: \n",df.isna().sum()) #Valores nulos
print("\n\nInfo:\n",df.info) #Información de dataframe
print("\n\nTipos:\n",df.dtypes) #Tipos de datos
print("\n\nValores únicos:\n",df.nunique()) #valores únicos

#%%%% Información de las variables
'''
entidadfederativa: Entidad federativa
sujetoobligado: Secretaría, unidad, o Instituto al que pertenecen
nombre: Nombre completo
denominacion, cargo: Puesto que desempeña
montoneto: Salario Neto
area: Plantel, oficina, dirección en la que trabajan
montobruto: Salario bruto
idInformacion: identificador
periodoreportainicio: fecha de inicio
periodoreportafin: fecha de fin
'''
#%%%% Gráficas y conteo de valores

valores_sujetoobligado = df['sujetoobligado'].value_counts()
print(valores_sujetoobligado,"\n\n")

valores_cargo = df['cargo'].value_counts()
print(valores_cargo,"\n\n")

valores_area = df['area'].value_counts()
print(valores_area,"\n\n")

sns.countplot(data=df, x="entidadfederativa")
plt.title('Distribución de la entidad federativa')
plt.xlabel('Categoría')
plt.ylabel('Frecuencia')
plt.xticks(rotation=90)
plt.show()

sns.histplot(df['montobruto'], kde=True)
plt.title('Distribución del salario bruto')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.show()

sns.histplot(df['montoneto'], kde=True)
plt.title('Distribución del salario neto')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.show()

x = df['montobruto']
y = df['montoneto']
plt.scatter(x, y)
plt.xlabel('Salario Bruto')
plt.ylabel('Salario Neto')
plt.title('Relación entre Salario Bruto y Salario Neto')
plt.show()
