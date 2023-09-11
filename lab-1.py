import pandas as pd
import numpy as np

path = "D:/Curso ibm/Data analysis with python/automovil-dataset.csv"

df = pd.read_csv(path,header=None)
print("The first 5 rows of the dataframe")
print(df.head(5))

#------------------------------Ejercicios---------------------------------

#Question #1: Check the bottom 10 rows of data frame "df".

print(df.tail(10))
#**********************************Fin de la pregunta 1*******************
#Agregar headers

headers= ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
print("headers\n",headers)

df.columns = headers
print("\n",df.head(5))

#podemos reemplazar el símbolo "?" con la palabra NaN con el método dropna() podemos remover los valores faltantes
df1 = df.replace('?',np.NaN)

#podemos eliminar los valores faltantes de la columna "price" de la siguiente manera

df= df1.dropna(subset=["price"], axis=0)
print(df.head(20))

#Pregunta 2, encuentra el nombre de las columnas del dataframe
print(df.columns,"\n")

#*******************Fin de la pregunta 2**************************************

#guardar un dataset, panda nos permite guardar dataset en formato csv usando el método dataframe.to_csv(), puedes agregar un filepath y un nombre dentro de signos de exclamación dentro del bracket
#por ejemplo si deseas guardar el dataframe df como automovil.csv en tu maquina local, puedes usar el syntax escrito mas abajo, donde index = False significa que la fila nombres no sera escrita

df.to_csv("automovil.csv", index=False)

#También podemos leer y guardar en otros formatos, podemos utilizar funciones similares como pd.read_csv() y df.to_csv()

print(" Data Formate        Read                Save")
print(" csv             pd.read_csv()       df.to_csv()")
print(" json            pd.read_json()      df.to_json()")
print(" excel           pd.read_excel()     df.to_excel()")
print(" hdf             pd.read_hdf()       df.to_hdf()")
print(" sql             pd.read_sql()       df.to_sql()")

print("Data types, los principales tipos almacenados en pandas dataframe son object, float, int, bool an datetime64, para poder entender de mejor manera acerca de cada atributo es bueno saber que tipo de dato contiene cada columna para eso usaremos el siguiente método\n")
print("\n",df.dtypes,"\n")

print("Descripción, si quisiéramos obtener un sumario estático de cada columna ejemplo 'count' significa valor, la columna desviación estándar, etc, usaremos el método describe como se detalla en el código a continuación: dataframe.describe()")

print(df.describe(),"\n")
print("este método retornara varios sumarios estáticos de tipo numérico, excluyendo NaN(Not a Number)")

print("Ahora que pasa si queremos incluir todas las columnas incluyendo aquellas que son tipo object?, para eso podemos incluir el siguiente argumento(include = all)")
print(df.describe(include="all"))

# Pregunta 3
print('Puedes seleccionar columnas del dataframe indicando el nombre de cada columna, por ejemplo\ndataframe[["column 1","column 2","column 3"]]\ndonde "column" es el nombre de la columna\ndataframe[["column 1","column 2","column 3"]].describe()\naplica el método a ".describe()" a las columnas "length" y "compression-ratio"')

df[["length","compression-ratio"]].describe()

print("\ninformación otro método que puedes usar para revisar tu dataset es dataframe.info() te entregara un sumario conciso de tu dataframe, este método imprime información acerca del dataframe incluyendo index dtype y columnas, valores non-null y memoria usada")
print(df.info())