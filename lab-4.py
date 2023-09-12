# Objetivos después de completar este lab seras capas de desarrollar modelos predictivos
# En esta sección, desarrollaremos varios modelos que predecirán el precio de car usando variables o caracteristicas. Esto es solo un estimado, pero debería darnos una idea objetiva de cuanto debe costar un auto
# Algunas preguntas que queremos preguntar en este modulo
# ¿Se si el vendedor me esta ofreciendo un precio justo por mi intercambio?
# ¿Se si he puesto un precio justo a mi auto?

# En data análisis, normalmente usamos Model Development para ayudarnos  a predecir futuras observaciones desde la información que tenemos. Un modelo nos ayudara a entender la relación exacta entre dos variables diferentes y como estas variables son usadas para predecir el resultado

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LinearRegression

filename = "D:/Curso ibm/Data analysis with python/automobileEDA.csv"
df = pd.read_csv(filename,header=0)
print(df.head(0))

# 1. Regresión lineal y Regresión lineal multiple (Linear regression and multiple linear regression)
# Regresión lineal (linear regression) Un ejemplo de Modelo de información que estaremos usando

# Regresión lineal simple (simple linear regression) es un método que nos ayuda a entender la relación entre dos variables:
# El predictor/ variable independiente(X)
# La respuesta/ variable dependiente(la que queremos predecir)(Y)

# El resultado de la Regresión lineal es una función lineal que predice la respuesta variable (dependiente) como una función de la variable predictor(independiente)

# Función lineal Y: Respuesta variable
#                X: Variable predictora
#                Yhat: a+bX

# A se refiere a la intersección de la regresión lineal, en otras palabras: El valor de Y cuando X es 0

# B se refiere al pendiente de la regresión lineal, en otras palabras: El valor de Y cuando X incremente 1 unidad

# Creando el objeto regresión lineal

lm = LinearRegression()
print(lm)