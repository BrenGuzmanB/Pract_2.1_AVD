# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 00:12:30 2023

@author: Bren Guzmán, Brenda García, María José Merino
"""

#%% LIBRERÍAS
import pandas as pd
from Preprocessing import find_optimal_lambda, box_cox_transform





#%% CARGAR ARCHIVO

df = pd.read_csv('salariesSample.csv')


#%% EXPLORACIÓN

#%%%% TIPOS DE DATOS, CONTEO DE VALORES NULOS, ETC
print("\n\nDescribe: \n",df.describe()) #estadísticos básicos
print("\n\n NaN Values: \n",df.isna().sum()) #Valores nulos
print("\n\nInfo:\n",df.info) #Información de dataframe
print("\n\Tipos:\n",df.dtypes) #Tipos de datos
print("\n\nValores únicos:\n",df.nunique()) #valores únicos

#%%%% INFORMACIÓN DE LAS VARIABLES
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
#%%%% 



