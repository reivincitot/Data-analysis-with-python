import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Para descargar el archivo que usaremos en este lab puedes usar este link: "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv\"

# Cargaremos la información y la guardaremos en el dataframe df:
filename = "D:/Curso ibm/Data analysis with python/automobileEDA.csv"
df = pd.read_csv(filename, header=0)
print(df.head(),"\n")

# Analizando patrones individuales usando visualización. Importaremos los paquetes "Matplotlib" y "Seaborn"

# Como escoger el método correcto de visualización?

print(df.dtypes, "\n")

# Pregunta 1. Cual es el tipo de dato en la columna "peak-rpm"?

print("El tipo de dato en la columna \"peak-rpm\" es:", df["peak-rpm"].dtypes, "\n")

# por ejemplo podemos calcular la correlación entre las variables de tipo "int64" o "float64" usando el método "corr":

# Seleccionando solo las columnas numéricas
numeric_columns = df.select_dtypes(include=['number'])

correlation_matrix = numeric_columns.corr()

print(correlation_matrix,"\n")

# Los elementos en diagonal son siempre uno; Estudiaremos la correlación, Pearson correlation para ser mas precisos al final del notebook. (el lab esta dentro de un jupyter notebook)

# Pregunta 2 Encuentra la correlación entre las siguientes columnas: bore, stroke, compression-ratio y horsepower

correlation_matrix = df[["bore","stroke","compression-ratio","horsepower"]].corr()

print(correlation_matrix,"\n")

# Variables numéricas continuas: Las variables numéricas continuas son variables que pueden contener cualquier valor en algún rango. ellas pueden ser de tipo "int64" o "float64". Una gran manera de visualizar estas variables is usando scatterplots en las lineas correspondientes.

# Para poder entender la relación linear  entre un variable individual y el precio, podemos usar "regplot" que trama el scatterplot más la regresión lineal precisa para la información

# Veamos varios ejemplos

# Relación linear positiva (Positive linear Relationship) Encontraremos el scatterplot de "engine-size" y "precio".

sns.regplot(x="engine-size",y="price", data= df)
print(plt.ylim(0,),"\n")
#plt.show()  #Descomentar para poder visualizar el gráfico

# A medida que "engine-size" sube, también lo hace el precio, esto indica una correlación positiva y directa entre estas dos variables. "engine-size" parece ser un buen predictor de precio dado que la regresión lineal es casi una diagonal perfecta

# Podemos examinar la correlación entre "engine-size" y price y ver que es aproximadamente 0.87.

df[["engine-size", "price"]].corr()

# "highway-mpg" es una variable potencial predictor de "price". vamos a encontrar el scatterplot de "highway-mpg" y precio

sns.regplot(x="highway-mpg", y="price", data=df)
print(plt.ylim(0,),"\n")
#plt.show()  #Descomentar para poder visualizar el gráfico

# Entre mas alto el valor de "highway-mpg" el valor de precio cae: esto indica una relación inversa/negativa entre estas dos variables. highway mpg puede ser potencialmente un predictor de price.

# Podemos examinar la correlación entre "highway-mpg" y "price" and see su aproximación -.704

df[["highway-mpg", "price"]].corr()

# Relación lineal débil veamos si "peak-rpm" es un predictor de variable de "price"

sns.regplot(x="peak-rpm", y="price", data=df)
print(plt.ylim(0,),"\n")
#plt.show()  #Descomentar para poder visualizar el gráfico

# "peak-rpm" no parece ser un buen predictor de "price" ya que la regresión lineal es casi horizontal, ademas los puntos de información esta bastante dispersos y lejos de la linea ajustada, mostrando mucha variabilidad. De ahi que no es una variable confiable.

# Podemos examinar la correlación entre "peak-rpm" y "price" y ver ue es aproximadamente -0.101616

df[["peak-rpm","price"]].corr()

# Pregunta 3a encuentra la correlación entre x="stroke" y y="price"

df[["stroke", "price"]].corr()

# Pregunta 3b dado los resultados de arriba esperas tu que haya una relación lineal entre "price" y "stroke" verifica tus resultados usando la función "regplot()"

sns.regplot(x="stroke", y="price", data=df)
plt.ylim(0,)
#plt.show()  #Descomentar para poder visualizar el gráfico

# Variables categóricas
# estas son variables que describen a una característica de una unidad de información, y son seleccionadas de un grupo pequeño de categorías. Las variables categóricas pueden tener el tipo "object" o "int64". Una buena manera de visualizar variables categóricas es usando boxplots

sns.boxplot(x="body-style", y = "price", data=df)
plt.ylim(0,)
#plt.show()  #Descomentar para poder visualizar el gráfico

# Podemos ver que la distribución de "price" entre los diferentes "body-style" se montan significativamente, asi que "body-style" no seria un buen predictor, examinemos "engine- location" y "price"

sns.boxplot(x="engine-location", y="price", data=df)
plt.ylim(0,)
#plt.show()  #Descomentar para poder visualizar el gráfico

# Acá podemos ver que la distribución de precio entre estas 2 posiciones del motor, frontal o trasera son lo suficiente mente distintas para tomar a "engine-location" como un potencial buen predictor de "price"

# Examinemos "drive-wheels" y "price"

sns.boxplot(x="drive-wheels", y="price", data=df)
plt.ylim(0,)
#plt.show()  #Descomentar para poder visualizar el gráfico

# Acá podemos ver que la distribución de precios entre las diferentes categorías de "drive-wheels", asi que podemos decir que drive-wheels puede ser un potencial predictor de price

# Análisis estadístico descriptivo (Descriptive Statistical Analysis)

# Primero miremos a las variables utilizando el método description la función describe automáticamente computa estadísticas básicas para todas las variables continuas. Cualquier valor NaN es saltado de forma automática

# Esto mostrara
# El contador de esa variable (count of the variable)
# El promedio (the mean)
# La desviación estándar (std)
# El valor mínimo (the minimum value)
# El IQR (interquartile Range 25%,50% y 75%)
# El valor máximo (the maximum value)

# Podemos aplicar el métodos "describe" de la siguiente manera
print(df.describe(),"\n")

# El setting por default de "describe" se salta algunas variables de tipo "object". Podemos aplicar
print(df.describe(include=["object"]),"\n")

# Cuenta de valores cuenta de valores es una buena manera de entender cuantos unidades de cada característica/variable tenemos. Podemos aplicar el método "value_counts" en la columna "drive-wheels". No olvides que el método "value_counts" solo funciona en las series de pandas, no en los data frame de pandas. Como resultado, solo podemos un solo bracket df['drive-wheels'], no dos brackets df[['drive wheels']]

print(df['drive-wheels'].value_counts(),"\n")

# Podemos convertir la seria en un dataframe de la siguiente forma

print(df['drive-wheels'].value_counts().to_frame())

# Repetiremos los pasos de mas arriba para guardar los resultado en el dataframe "drive_wheels_counts" y renombrar la columna "drive-wheels"a "value_counts".

drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive_wheels':'value_counts'}, inplace=True)
print(drive_wheels_counts, "\n")

# Ahora renombraremos el index a "drive_wheels"

drive_wheels_counts.index.name = 'drive-wheels'

# Podemos repetir el proceso para "engine-location"

engine_loc_count = df['engine-location'].value_counts().to_frame()
engine_loc_count.rename(columns={'engine-location':'value_counts'}, inplace=True)
engine_loc_count.index.name = 'engine-location'
print(engine_loc_count.head(10),"\n")

# Luego de examinar el conteo de valores de engine location, podemos ver que engine location no seria un buena variable predictora de price. Esto es por que solo tiene tres vehículos con motor trasero y 198 con motor delantero, asi que este resultado es sesgado. Por lo tanto, no seremos capaces de llegar a ninguna conclusion a cerca del motor

# 4.Lo básico de agrupar. El método "groupby" agrupa información por diferentes categorías. La información base agrupada en una o muchas variables, y el análisis es llevado en grupos individuales
# Por ejemplo, agruparemos poro la variable "drive-wheels". Podemos ver que hay tres diferentes categorías de "drive-wheels".

print(df['drive-wheels'].unique())

# Si queremos saber, en promedio que tipo de ruedas es tiene mas valor o es mas valorada, podemos agrupar "drive-wheels" y luego promediar-las

# Podemos seleccionar la columna "drive-wheels", "body-style" y "precio" y luego asignarlo a la variable "df_group_one"

df_group_one = df[['drive-wheels','body-style', 'price']]

# Ahora podemos calcular el promedio de price por cada una de las categorías

df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False)['price'].mean()
print("\n",df_group_one)

# Desde nuestra información , parece que rear-wheel son los vehículos que en promedio son mas costosos, mientras que 4-wheel son aproximadamente el mismo precio

# También puedes agrupar por varias variables. Por ejemplo agruparemos por los dos "drive-wheels" y "body-style". Esto agrupa el dataframe por la combinación única de "drive-wheel" y "body-style", Almacenaremos los resultados en la variable "group_test1"

df_gptest= df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False)['price'].mean()
print("\n",grouped_test1,"\n")

# Esta información agrupada es mas fácil de visualizar cuando la traspasas a una tabla de pivot (pivot table). Una tabla de pivot es como una hoja de Excel, con una variable a lo largo de la columna y otra a lo largo de la fila. Podemos convertir el dataframe en una tabla de pivot usando el método "pivot" para crear una tabla de pivot desde los grupos

# En este caso dejaremos "drive-wheels" como la fila de la tabla y "body-style" como columna y pivot de la tabla:

grouped_pivot = grouped_test1.pivot(index= 'drive-wheels', columns='body-style')
print(grouped_pivot)

# Usualmente, no tenemos información de algunas celdas pivot. Podemos llenar estas celdas con el valor 0, pero cualquier otro valor puede potencialmente ser usado. Debe ser mencionado que la información faltante es un tópico bastante complejo, y es un curso completo por si mismo.

grouped_pivot = grouped_pivot.fillna(0)
print(grouped_pivot)

# Pregunta 4 usa la función "groupby" para encontrar el promedio "price" de cada vehículo basado en "body-style"

df_gptest2=df[['body-style', 'price']]
grouped_test2 = df_gptest2.groupby(['body-style'],as_index=False)['price'].mean()
print(grouped_test2,"\n")