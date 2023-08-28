import pandas as pd
import numpy as np
import matplotlib as plt

# indicando donde se encuentra la información guardada
path = "D:/Curso ibm/Data analysis with python/automovil-dataset.csv"

# creando headers para el data frame con el que se trabajara
headers1 = "symboling normalized-losses make fuel-type aspiration num-of-doors body-style drive-weels engine-location weel-base length width height curb-weight engine-type num-of-cylinders engine-size fuel-system bore stroke compression-ratio horsepower peak-rpm city-mpg, highway-mpg price"

# Separando los headers para que sean asignados a cada columna
headers = headers1.split(" ")

# Asignando un variable al nombre del archivo
filename = "automovil-dataset.csv"

# creando el dataframe en donde juntamos el nombre del archivo con los headers creados mas arriba
df = pd.read_csv(filename, names=headers)

print('\nImprimiendo las 5 primeras filas del dataframe con sus respectivos headers\n')
print(df.head())

print('\ncomo podemos ver en el dataframe hay varias filas en el que se ve el singo "?", estos son missing values los cuales dificultaran nuestro análisis a futuro. Así que como lidiamos con la información faltante: pasos para trabajar con información faltante 1) identificar la información faltante 2) encargarse de la informacion faltante 3) corregir los formatos de la información')

print('\nIdentificar y manejar valores faltante "Missing values" convirtiendo "?" a NaN,en el data set car, la información faltante viene como "?". Reemplazaremos "?" con NaN(Not a Number) acá usaremos  la función .replace(A, B, inplace = True)')

df.replace("?", np.nan, inplace=True)
print(df.head(5))

print("\nEvaluación para información faltante, los valores faltantes son convertidos por default. Usaremos las siguientes funciones para identificar estos valores faltantes. hay dos metodos para detectar los valores faltantes 1) isnull()  2) notnull() la salida es un valor booleano si el valor pasado es efectivamente un valor faltante")

missing_data = df.isnull()
print(missing_data.head(5))

print('\n"True" significa que el valor en la columna es un valor faltante, si el retorno es False significa que el valor no es un valor faltante')

print('\nContar los valores faltantes en cada columna, usando un loop , podemos rápidamente darnos cuenta la cantidad de valores faltantes en cada columna, como mencionamos arriba  "True" representa un valor faltante y "False" significa que la column contiene un valor valido en el cuerpo de un loop for usaremos el metodo ".values_counts()" para contar el numero de valores "True"')

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

print('Podemos ver que las cuatro puertas son el tipo mas común. Tambien podemos usar el metodo ".idemax()" para calcular el tipo mas comun')
print(df["num-of-doors"].value_counts().idxmax())
print('\nEl procedimiento de reemplazo es el mismo')
df['num-of-doors'].replace(np.nan,"four", inplace=True)

print('Finalmente desecharemos las columnas que no tienen ningún valor')
df.dropna(subset=["price"], axis=0, inplace=True)

df.reset_index(drop=True, inplace=True)

print(df.head())
print('\nCorregir el formato de la información')
