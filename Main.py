
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

#%% PREPROCESAMIENTO
#%%% Outliers

# histogramas previo a eliminar los valores atípicos
sns.histplot(df['montobruto'], kde=True)
plt.title('Distribución del salario bruto (prev)')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.show()

sns.histplot(df['montoneto'], kde=True)
plt.title('Distribución del salario neto (prev) ')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.show()

#FILTROS
outliersmb = two_sigma(df, 'montobruto')    #Filtro 2 sigma
common_indices = df[df.isin(outliersmb.to_dict(orient='list')).all(axis=1)].index
df_cleaned = df.drop(common_indices)
df_cleaned = filter_box_cox(df_cleaned, "montoneto", range=[-2.5, 2.5])    #filtro box-cox

# Histogramas posterior a eliminar los outliers
sns.histplot(df_cleaned['montobruto'], kde=True)
plt.title('Distribución del salario bruto (post)')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.show()

sns.histplot(df_cleaned['montoneto'], kde=True)
plt.title('Distribución del salario neto (post) ')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.show()

#%%% Imputación

registros_nulos = df_cleaned[df_cleaned["montobruto"].isnull()]
indices_nulos = df_cleaned[df_cleaned["montobruto"].isnull()].index

df_cleaned = impute_median(df_cleaned, "montobruto")    #Imputación mediana usando los intercuartiles
registros_imputados = df_cleaned.loc[indices_nulos]
#%%% Transformación

lmbda = find_optimal_lambda(df_cleaned['montobruto'])
df_cleaned['montobrutoT']= box_cox_transform(df_cleaned['montobruto'], lmbda)

# Histogramas posterior a la transformación
sns.histplot(df_cleaned['montobrutoT'], kde=True)
plt.title('Distribución del salario bruto (post box-cox transform)')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.show()


lmbda = find_optimal_lambda(df_cleaned['montoneto'])
df_cleaned['montonetoT']= box_cox_transform(df_cleaned['montoneto'], lmbda)

# Histogramas posterior a la transformación
sns.histplot(df_cleaned['montonetoT'], kde=True)
plt.title('Distribución del salario neto (post box cox transform)')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.show()

#%%% Normalización

df_cleaned = min_max(df_cleaned, "montobrutoT")

# Histogramas posterior a la normalización
sns.histplot(df_cleaned['montobrutoT'], kde=True)
plt.title('Distribución del salario bruto (post min-max normalize)')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.show()

df_cleaned = min_max(df_cleaned, "montonetoT")

# Histogramas posterior a la normalización
sns.histplot(df_cleaned['montonetoT'], kde=True)
plt.title('Distribución del salario neto (post min-max normalize)')
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.show()

#%%% Codificación

df_cleaned = one_hot_encoding(df_cleaned, "entidadfederativa")

columnas_a_eliminar = ['sujetoobligado', 'nombre', 'denominacion', 'montoneto', 'cargo',
       'area', 'montobruto', 'idInformacion', 'periodoreportainicio',
       'periodoreportafin']
df_cleaned = df_cleaned.drop(columns=columnas_a_eliminar)

#%% REGRESIÓN
#%%%% Datos preprocesados
# Se va a estimar el salario neto
X = df_cleaned.drop("montonetoT", axis=1)  # Características
y = df_cleaned["montonetoT"]  # Variable objetivo

# Conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# entrenamiento del modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Test
y_pred = model.predict(X_test)

# Evaluación del modelo
mseP = mean_squared_error(y_test, y_pred)
r2P = r2_score(y_test, y_pred)


#%%% Datos originales

# Se va a estimar el salario neto

df = df[['montoneto', 'montobruto']]
df = df.dropna()

X = df.drop("montoneto", axis=1)  # Características
y = df["montoneto"]  # Variable objetivo

# Conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# entrenamiento del modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Test
y_pred = model.predict(X_test)

# Evaluación del modelo
mseO = mean_squared_error(y_test, y_pred)
r2O = r2_score(y_test, y_pred)


#%%% Resultados

print("\nResultados con los datos originales: \n")
print(f"Error Cuadrático Medio (MSE): {mseO}")
print(f"Coeficiente de Determinación (R^2): {r2O}\n")

print("\nResultados con los datos preprocesados: \n")
print(f"Error Cuadrático Medio (MSE): {mseP}")
print(f"Coeficiente de Determinación (R^2): {r2P}\n")
