import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# indicando donde se encuentra la información guardada
path = "D:/Curso ibm/Data analysis with python/automovil-dataset.csv"

# creando headers para el data frame con el que se trabajara
headers1 = "symboling normalized-losses make fuel-type aspiration num-of-doors body-style drive-weals engine-location weal-base length width height curb-weight engine-type num-of-cylinders engine-size fuel-system bore stroke compression-ratio horsepower peak-rpm city-mpg highway-mpg price"

# Separando los headers para que sean asignados a cada columna
headers = headers1.split(" ")

# Asignando un variable al nombre del archivo
filename = "automovil-dataset.csv"

# creando el dataframe en donde juntamos el nombre del archivo con los headers creados mas arriba
df = pd.read_csv(filename, names=headers)

print('\nImprimiendo las 5 primeras filas del dataframe con sus respectivos headers\n')

print(df.head())

print('\ncomo podemos ver en el dataframe hay varias filas en el que se ve el singo "?", estos son missing values los cuales dificultaran nuestro análisis a futuro. Así que como lidiamos con la información faltante: pasos para trabajar con información faltante 1) identificar la información faltante 2) encargarse de la información faltante 3) corregir los formatos de la información')

print('\nIdentificar y manejar valores faltante "Missing values" convirtiendo "?" a NaN,en el data set car, la información faltante viene como "?". Reemplazaremos "?" con NaN(Not a Number) acá usaremos  la función .replace(A, B, inplace = True)')

df.replace("?", np.nan, inplace=True)

print(df.head(5))

print("\nEvaluación para información faltante, los valores faltantes son convertidos por default. Usaremos las siguientes funciones para identificar estos valores faltantes. hay dos métodos para detectar los valores faltantes 1) isnull()  2) notnull() la salida es un valor booleano si el valor pasado es efectivamente un valor faltante")

missing_data = df.isnull()

print(missing_data.head(5))

print('\n"True" significa que el valor en la columna es un valor faltante, si el retorno es False significa que el valor no es un valor faltante')

print('\nContar los valores faltantes en cada columna, usando un loop , podemos rápidamente darnos cuenta la cantidad de valores faltantes en cada columna, como mencionamos arriba  "True" representa un valor faltante y "False" significa que la column contiene un valor valido en el cuerpo de un loop for usaremos el método ".values_counts()" para contar el numero de valores "True"')

for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")

print('\nBasado en el sumario de arriba cada columna tiene 205 filas y 7 columnas contienen valores faltante "missing values"')
print('1. "normalized-losses": 41 missing data')
print('2. "num-of-doors": 2 missing data')
print('3. "bore": 4 missing data')
print('4. "stroke": 4 missing data')
print('5. "horsepower": 2 missing data')
print('6. "peak-rpm": 2 missing data')
print('7. "price": 4 missing data\n')
print("\nManejando Missing data")
print("1. Drop data")
print("    a. desechar la fila completa")
print("    b. desechar la columna completa")
print("2. reemplazar información")
print("    a. reemplazarla por significado")
print("    b. reemplazarla por frecuencia")
print("    c. reemplazarla basadas en otras funciones\n")

print("Eliminar la columna completa solo sera usada en el caso de que a la columna le falten la mayoría de los valores, ninguna de la columnas cumple este parámetro así que contamos con completa libertad para elegir cualquier otro método de reemplazo")

print("\nReemplazando por el método mean: ")
print('"normalized-losses": 41 valores perdidos, reemplazarlo con mean ')
print('"stroke": 4 valores perdidos, reemplazarlos con mean')
print('"bore": 4 valores perdidos, reemplazarlos con mean')
print('"horsepower": 2 valores perdidos, reemplazarlos con mean')
print('"peak-rpm": 2 valores perdidos, reemplazarlos con mean')

print('\nReemplazar por frecuencia\n')
print('"num-of-doors": 2 valores perdidos, reemplazarlos con mean\n     Razón: 84% de los sedans es de 4 puertas, dado que 4 puertas es lo mas frecuente, es mas probable que asi sea.\n')


print('Eliminar la columna completa\n')
print('"price":4 valores perdidos, reemplazarlos con mean\n   Razón: precio es lo que queremos predecir')

print('\nCalculando el valor Medio (mean) para la columna "normalized-losses" ')

avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)

print('\nAverage of normalized-losses: ', avg_norm_loss)

print('\nRemplazando "NaN" con el valor medio en la columna "Normalized-losses"')

df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)


print('\nCalculando el valor medio para la columna "bore"')

avg_bore = df["bore"].astype("float").mean(axis=0)

print('Average of bore: ', avg_bore)

print('\nReemplazando "NaN" con el valor medio de la columna "bore"')

df['bore'].replace(np.nan, avg_bore, inplace=True)


# Pregunta 1, basado en los ejemplos anteriores , reemplaza NaN in la columna "stroke" con el valor medio

avg_stroke = df['stroke'].astype('float').mean(axis=0)

print('\nAverage of stroke :', avg_stroke)

print('\nReemplazando "NaN" con el valor medio de la columna "stroke"')

print(df['stroke'].replace(np.nan, avg_stroke, inplace=True))

# Calcula el valor medio de la columna "horsepower"
avg_horsepower = df["horsepower"].astype("float").mean(axis=0)

print('\nAverage of horsepower: ', avg_horsepower)

print('\nReemplazando "NaN" con el valor medio de la columna "horsepower"')

df["horsepower"].replace(np.nan, avg_horsepower, inplace=True)

# Calcula el valor medio de la columna "Peak-rpm"
avg_peak_rpm = df["peak-rpm"].astype("float").mean(axis=0)

print("\nAverage peak-rpm:", avg_peak_rpm)

print('\nReemplazando "NaN" con el valor medio de la columna "peak-rpm"')

df["peak-rpm"].replace(np.nan, avg_peak_rpm, inplace=True)

print('\nPara ver que valores están presentes en cada columna podemos usar el método ".value_counts()"')

print(df["num-of-doors"].value_counts())

print('Podemos ver que las cuatro puertas son el tipo mas común. También podemos usar el método ".idxmax()" para calcular el tipo mas común')

print(df["num-of-doors"].value_counts().idxmax())

print('\nEl procedimiento de reemplazo es el mismo')

df['num-of-doors'].replace(np.nan, "four", inplace=True)

print('Finalmente desecharemos las columnas que no tienen ningún valor')

df.dropna(subset=["price"], axis=0, inplace=True)

df.reset_index(drop=True, inplace=True)

print(df.head())

print('\nCorregir el formato de la información')
print('\nel ultimo paso en limpiar la información es asegurarnos que toda la información tenga el formato correcto (int,float,text u otro), en panda usamos: \n.dtype() para verificar el tipo de dato\n.astype() para cambiar el tipo de dato\n')

print(df.dtypes)

print('\nComo podemos ver arriba, algunas columnas no tiene el tipo correcto de dato.Variables numéricas deberían tener el tipo "float" o "int", variables con string deberían tener la categoría object, por ejemplo: "bore" y "stroke" son valores numéricos que describen el motor, asi que deberíamos esperar un tipo "float" o un tipo "int" pero son mostrados como tipo "object". Tenemos que convertir esta información al tipo correcto de cada columna usando el método "astype()"')

df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")

df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")

df[["price"]] = df[["price"]].astype("float")

df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

print("\n", df.dtypes)

print('Estandarización de la información. La información es usualmente recolectada por diferentes agencias en diferentes formatos (Data standardization también es un termino para un normalización de un tipo de data donde restamos el promedio (mean) y dividimos por la desviación estándar (standard deviation)\n')

print('Que es la estandarización?\nEstandarización es el proceso de transformar la información a un formato común, permitiendo le a los investigadores hacer comparaciones significativa\n')

print('Ejemplo\nTransforma mpg (miles per gallon) a L/100km:')

print('En nuestro data set las columnas de consumo de combustible son "city-mpg" y "highway-mpg" son presentadas como la unidad de medida mpg. Asumiendo que desarrollamos una aplicación que acepte como medida L/100km como estándar.\nNecesitaremos aplicar la transformación de la información de mpg a L/100km.\La formula de conversion es:\n L/100km = 235/mgp.\nPodemos hacer una gran variedad de operaciones matemáticas directamente en pandas')

print(df.head())

# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df["city-L/100km"] = 235 / df["city-mpg"]

df.rename(columns={'city-mpg': 'city-L/100km'})

# check your transformed data
print(df.head())

# Pregunta 2 de acuerdo al ejemplo de arriba , transformamos la columna mpg a L/100km
df["highway-L/100km"] = 235 / df["highway-mpg"]

# renombrando la columna highway-mpg a highwayL/100km
df.rename(columns={'highway-mpg': 'highway-L/100km'}, inplace=True)
print(df.head())

print('Normalización de la información')

print('Por que normalizar.\nNormalizar es el proceso de transformación de un numero de variables a un rango similar.\nTípicamente normalizar incluye escalar la variable para que el promedio de la variable sea 0, escalar la variable para que la variación sea 1, o escalar la variable para que el valor valla desde el rango 0 a 1.\n')

print('Para demostrar la normalización, vamos a escalar las columnas "length","width" y "height".\n')
print('Objetivo: Normalizar las variables para que el valor de sus rangos vallan de 0 a 1.\nAcercamiento: Reemplazar el valor original por (original value)/(maximum value)')

df["length"] = df["length"]/df["length"].max()

df["width"] = df["width"]/df["width"].max()

# Pregunta 3 de acuerdo al ejemplo de arriba normaliza la columna height.
df["height"] = df["height"]/df["height"].max()

print("df[\"height\"]= df[\"height\"]/df[\"height\"].max()")
print("df[[\"length\",\"width\",\"height\"]].head()")

print(df[["length", "width", "height"]].head())
print('\nAcá podemos ver normalizadas las columnas "length","width", "height" en el rango de 0 a 1\n')

print("Binning\nBinning es el proceso de transformar continuamente numerosas variables y agruparlas en un contenedor \"bins\"")

print("Ejemplo\nEn nuestro dataset \"horsepower\" es un valor real una variable que va entre los rangos 48 a 288 y tiene 59 valores únicos, que sucedería si queremos saber el rango de precio entre autos con muchos caballos de fuerza, una cantidad media de caballos de fuerza y aquellos que poseen pocos caballos de fuerza(3 tipos), podemos arreglarlos en tres diferentes contenedores o \"bins\" para simplificar el análisis\nUsaremos el método cut de pandas para segmentar en 3 la columna \"horsepower\"")

print("Ejemplo de Binning en pandas\nConvertir la información al tipo correcto")

df["horsepower"] = df["horsepower"].astype(int, copy=True)
print("df[\"horsepower\"] = df[\"horsepower\"].astype(int, copy=True)\n")

print("Vamos a trazar el histograma de \"horsepower\" para ver como luce su distribución")

plt.hist(df["horsepower"])
#agregar marcadores x/y y el titulo del trazado
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepowers bins")
plt.show()

print("plt.hist(df[\"horsepower\"])\n#Agregando marcadores x/y y el titulo\plt.xlabel(\"horsepower\")\nplt.ylabel(\"count\")\nplt.title(\"horsepower bins\")\nplt.show()")




"Queremos tres contenedores de igual tamaño y ancho asi que usaremos numpy linspace(start_value, end_value, numbers_generated function.\nDado que queremos incluir el valor mínimo de horsepower, definiremos como start_value = min(df[\"horsepower\"])\nDado que queremos incluir el valor máximo de horsepower, definiremos como end_value = max(df[\"horsepower\"])\nDado que queremos 3 bins de igual largo, deben haber 4 divisores, asi que numbers_generated = 4\nCrearemos un bin array con un valor mínimo usando el ancho de banda calculado arriba. Los valores determinaran cuando un bind termina y cuando otro comienza.)"

bins=np.linspace(min(df["horsepower"]),max(df["horsepower"]),4)
print(bins)

print("bins=np.linspace(min(df[\"horsepower\"]),max(df[\"horsepower\"]),4\nprint(bins)")

#Definiendo el nombre de los grupos
print("Definiendo el nombre de los grupos")

group_names = ['Low', 'Medium', ' High']

print("group_names = [\"Low\", \"Medium\,\"High\"]")

#Aplicaremos la función "cut" para definir cada valor de df["horsepower"] pertenece.
print("Aplicaremos la función \"cut\" para definir cada valor de df[\"horsepower\" pertenece.]")

df["horsepower-binned"]= pd.cut(df["horsepower"], bins, labels= group_names,include_lowest=True)

df[["horsepower","horsepower-binned"]].head(20)

#Veamos el numero de vehículos en cada contenedor (bins)
print("Veamos el numero de vehículos en cada contenedor (bins)")

df["horsepower-binned"].value_counts()

print("Tracemos la distribución de cada contenedor (bins)")

plt.bar(group_names,df["horsepower-binned"].value_counts())
print("plt.bar(group_names,df[\"horsepower-binned\"].value_counts()")

plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")
plt.show()

print("plt.xlabel(\"horsepower\")\nplt.ylabel(\"count\")\nplt.title(\"horsepower bins\")\nplt.show")

print("Mira el dataframe arriba cuidadosamente. Encontraras que la ultima columna entrega el contenedor (bins) para \"horsepower\"basado en tres categorías (\"Low\", \"Medium\"),\"High\".\nAcotamos exitosamente el intervalo desde 59 a 3")

print("Visualización de los contenedores (bins)\n")
print("Normalmente, un histograma es usado para visualizar la distribución de los contenedores (bins) que creamos arriba")

plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("Horsepower bins")

print("plt.xlabel(\"horsepower\")\nplt.ylabel(\"count\")\nplt.title(\"Horsepower bins\")")