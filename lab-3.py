import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Para descargar el archivo que usaremos en este lab puedes usar este link: "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv\"

# Cargaremos la información y la guardaremos en el dataframe df:
filename = "D:/Curso ibm/Data analysis with python/automobileEDA.csv"
df = pd.read_csv(filename, header=0)
print(df.head())

# Analizando patrones individuales usando visualización. Importaremos los paquetes "Matplotlib" y "Seaborn"

# Como escoger el método correcto de visualización?

print(df.dtypes, "\n")

# Pregunta 1. Cual es el tipo de dato en la columna "peak-rpm"?

print("El tipo de dato en la columna \"peak-rpm\" es:", df["peak-rpm"].dtypes, "\n")

# por ejemplo podemos calcular la correlación entre las variables de tipo "int64" o "float64" usando el método "corr":

# Seleccionando solo las columnas numéricas
numeric_columns = df.select_dtypes(include=['number'])

correlation_matrix = numeric_columns.corr()

print(correlation_matrix)

# Los elementos en diagonal son siempre uno; Estudiaremos la correlación, Pearson correlation para ser mas precisos al final del notebook. (el lab esta dentro de un jupyter notebook)

# Pregunta 2 Encuentra la correlación entre las siguientes columnas: bore, stroke, compression-ratio y horsepower

correlation_matrix = df[["bore","stroke","compression-ratio","horsepower"]].corr()

print(correlation_matrix)

# Variables numéricas continuas: Las variables numéricas continuas son variables que pueden contener cualquier valor en algún rango. ellas pueden ser de tipo "int64" o "float64". Una gran manera de visualizar estas variables is usando scatterplots en las lineas correspondientes.

#  Para poder entender