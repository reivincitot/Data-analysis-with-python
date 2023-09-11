import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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

# Variables: Drive Wheels y Body Style vs Price.
# Vamos a usar un mapa de calor para visualizar la relación entre Body Style y Price

plt.pcolor(grouped_pivot, cmap="RdBu")
plt.colorbar()
#plt.show()

# El mapa de calor traza la variable objetivo (precio) proporcional al color con respecto a las variables "rueda motriz" y "estilo de carrocería" en los ejes vertical y horizontal, respectivamente. Esto nos permite visualizar cómo se relaciona el precio con la "rueda motriz" y el "estilo de carrocería".

# Los marcadores por defecto no proporcionan ninguna información util para nosotros, Vamos a cambiar eso

fig, ax= plt.subplots()
im = ax.pcolor(grouped_pivot, cmap="RdBu")

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor= False)

# Rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
#plt.show()

# Visualizar es muy importante en data science, y los paquetes de visualización de Python proveen una gran libertad. Continuaremos en mas profundidad en un curso separado de Python.

# La pregunta principal que queremos contestar en este modulo es, "Cuales son las caracteristicas principales que afecta el precio de un vehículo"

# Para tener una mejor medida de la importancia de las caracteristicas, miramos a la correlación de estas variables con el precio del vehículo, en otras palabras: Como depende el precio de este vehículo con esta variable

# 5 Correlación y Causa
# Correlación: Una medida del grado de independencia entre variables
# Causation: La relación entre causa y efecto entre dos variables.
# Es importante saber la diferencia entre estos dos. Correlación no implica causa. Determinar correlación es mas fácil que determinar la causa, la causa puede requerir hacer experimentos de forma independiente

# Pearson Correlation. Pearson correlation mide la dependencia linear entre dos variables X y Y.
# El coeficiente resultante es un valor entre -1 y 1 incluso donde:
# 1: Correlación positiva perfecta
# 0: Correlación no linear, estas dos variables casi no se afectan
# -1: Correlación lineal negativa

# Pearson correlation es el método por defecto de la función "corr". Como antes, podemos calcular Pearson correlation de las variables  "int64" o "float64".

numeric_columns = df.select_dtypes(include=['number'])

correlation_matrix = numeric_columns.corr()

print(correlation_matrix,"\n")

# Algunas veces nos gustaría saber el significado del estimado de la correlación.

# P-value. Que es P-value? P-value es la probabilidad que la correlación entre estas dos variables tengan estadisticamente un significado. Normalmente, escogemos un significado de 0.05, lo que significa que estamos un 95% seguros o confiados que esa correlación entre las dos variables es significativa

# Por convención cuando
# P-value es < 0.001: decimos qye hay fuertes evidencias que la correlación es significativa.
# P-value es < 0.05: hay una evidencia moderada que la correlación es significativa
# P-value es 0.1: hay una evidencia débil que la correlación entre variable sea significativa
# Podemos obtener esta información usando el módulo "stats" de la librería "scipy".
# Wheel-Base vs Price
# Vamos a calcular el coeficiente Pearson Correlation y P-value de "wheel-base" y precio

pearson_coef, p_value = stats.pearsonr(df['wheel-base'],df['price'])
print("El coeficiente de Pearson correlation es: ", pearson_coef, "junto al P-value de P =",p_value,"\n")

print("Conclusión Dado que P-value es <0.001, la correlación entre wheel-base y price es estadisticamente significante, aunque la relación lineal no es extremadamente fuerte (~0.585)\n")

# Horsepower vs Price vamos a calcular Pearson correlation y P-value de "horsepower" y "price".

pearson_coef, p_value = stats.pearsonr(df['horsepower'],df['price'])

print("El coeficiente de Pearson correlation es: ", pearson_coef,"junto con el P-value de P = ",p_value,"\n")

print("Conclusion: Dado que P-value es < 0.001, la correlación entre horsepower y price es estadisticamente significativa, y la relación lineal entre es bastante fuerte (~0.809, cercano a 1)\n")

# Length vs Price, vamos a calcular Pearson correlation y P-value de "length" y "price".

pearson_coef, p_value = stats.pearsonr(df['length'],df['price'])

print("El coeficiente de Pearson correlation es: ", pearson_coef,"junto con el P-value de P = ",p_value,"\n")

print("Conclusion: Dado que P-value es < 0.001, la correlación entre length y price es estadisticamente significativa, y la relación lineal es moderadamente fuerte (~0.691)\n")

# Width vs Price, vamos a calcular Pearson correlation y P-value de "width" y "price".

pearson_coef, p_value= stats.pearsonr(df['width'],df['price'])

print("El coeficiente de Pearson correlation es: ", pearson_coef,"junto con el P-value de P = ",p_value,"\n")

print("Conclusion: Dado que P-value es < 0.001, la correlación entre width y price es estadisticamente significativa, y la relación lineal es bastante fuerte (~0.751)\n")

# Curb-weight vs Price, vamos a calcular Pearson correlation y P-value de "curb-weight" y "price"

pearson_coef, p_value= stats.pearsonr(df['curb-weight'],df['price'])

print("El coeficiente de Pearson correlation es: ", pearson_coef,"junto con el P-value de P = ",p_value,"\n")

print("Conclusion: Dado que P-value es < 0.001, la correlación entre curb-weight y price es estadisticamente significativa, y la relación lineal es bastante fuerte (~0.834)\n")

# Engine-size vs Price, vamos calcular Pearson correlation y P-value de "engine-size" y "price"

pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])

print("El coeficiente de Pearson correlation es: ", pearson_coef,"junto con el P-value de P = ",p_value,"\n")

print("Conclusion: Dado que P-value es < 0.001, la correlación entre engine-size y price es estadisticamente significativa, y la relación lineal es bastante fuerte (~0.872)\n")

# Bore vs Price, vamos a calcular el coeficiente Pearson correlation y P-value de "bore" y "price".

pearson_coef, p_value = stats.pearsonr(df['bore'],df['price'])

print("El coeficiente de Pearson correlation es: ", pearson_coef,"junto con el P-value de P = ",p_value,"\n")

print("Conclusion: Dado que P-value es < 0.001, la correlación entre bore y price es estadisticamente significativa, y la relación lineal es moderada (~0.521)\n")

# City-mpg vs Price vamos a calcular el coeficiente Pearson correlation y P-value de "city-mpg" y "price"

pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])

print("El coeficiente de Pearson correlation es: ", pearson_coef,"junto con el P-value de P = ",p_value,"\n")

print("Conclusion: Dado que P-value es < 0.001, la correlación entre city-mpg y price es estadisticamente significativa, y la relación lineal es negativa y moderadamente fuerte (-0.687)\n")

# Highway-mpg vs Price, vamos a calcular el coeficiente Pearson correlation y P-value de "highway-mpg" y "price"

pearson_coef, p_value = stats.pearsonr(df['highway-mpg'],df['price'])

print("El coeficiente de Pearson correlation es: ", pearson_coef,"junto con el P-value de P = ",p_value,"\n")

print("Conclusion: Dado que P-value es < 0.001, la correlación entre highway-mpg y price es estadisticamente significativa, y la relación lineal es negativamente y moderadamente fuerte (-0.705)\n")

# 6 ANOVA
# ANOVA : Análisis de la varianza (Analysis of Variance)
# El análisis de la varianza(ANOVA) es un método estadístico usado para probar si existen diferencias significativas entre las medidas de dos o mas grupos. ANOVA devuelve dos parámetros

# F-test score: ANOVA asume que la media de todos los grupos es la misma, calcula cuanto es que se desvía la media verdadera de esta presunción y lo reporta como F-test score.Un resultado grande significa que hay una gran diferencia entre los dos promedios

# P-value P-value te dice estadisticamente, cuan significativamente esta calculado nuestro valor

# Si nuestra variable precio es significativamente estrecha con la variable que analizamos, esperamos que ANOVA devuelva un gran F-test y un pequeño P-value.

# Drive wheels. Puesto que ANOVA analiza las diferencias entre diferentes grupos de la misma variable, la función groupby sera de gran ayuda. Puesto que ANOVA algorítmicamente promedia la información de forma automática

grouped_test2 = df_gptest[['drive-wheels','price']].groupby(['drive-wheels'])
print(grouped_test2.head(2),"\n")

print(df_gptest)

# Podemos obtener los valores del método group usando el método "get_group"

print(grouped_test2.get_group('4wd')['price'])

# Podemos usar la función "f_oneway" en el modulo "stats" para obtener el F-test score y P-value.

#ANOVA
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'],grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])
print("Los resultados ANOVA: F= ", f_val,", P= ",p_val)

# Este es un gran resultado con un resultado grande en F-test mostrando una correlación fuerte y un P-value casi 0 lo que implica una significación estadística casi segura. Pero, ¿significa esto que los tres grupos evaluados están tan altamente correlacionados?

# Examinemos las por separado

# fwd y rwd

f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'],grouped_test2.get_group('rwd')['price'])

print("Los resultados ANOVA para : F= ",f_val," P= ",p_val)

# 4wd y rwd

f_val,p_val= stats.f_oneway(grouped_test2.get_group('4wd')['price'],grouped_test2.get_group('rwd')['price'])

print("Los resultados ANOVA para F= ", f_val,"P= ",p_val)

# Conclusión : Variables importantes
# Ahora tenemos una mejor idea de como luce cada información y que variables son importantes para tomar en cuenta cuando predecimos el valor de un vehículo y las hemos disminuido a las siguientes variables

# Variables numéricas continuas (Continuous numerical variables):
# Length
# Width
# Curb-weight
# Engine-size
# Horsepower
# City-mpg
# Highway-mpg
# Wheel-Base
# Bore

# Variables categóricas (Categorical variables)

# Drive-wheels
# Ahora nos moveremos a construir un modelo de machine learning para automatizar el análisis, alimentando el modelo con las variables que significativamente afectan nuestra variable objetivo, lo que mejorara la performance de nuestro modelo de predicción