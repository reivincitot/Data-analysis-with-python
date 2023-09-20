# Después de completar este lab usted deberá ser capas de evaluar y refinar un modelo de predicción.

# Tabla de contenido:
# Evaluación del modelo
# Over-fitting, Under-fitting y Selección de Modelo
# Ridge Regression
# Grid Regression

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive,fixed,interact_manual
from sklearn.model_selection import train_test_split
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/module_5_auto.csv'
df = pd.read_csv(path)
df.to_csv("module_5_auto.csv")

# Primero solo usaremos información numérica

df = df._get_numeric_data()
print(df.head())

# Librerías para el entramado o plotting
# from ipywidgets import interact, interactive,fixed,interact_manual

# Funciones para el entramado o plotting
def DistributionPlot(Redfunction,Bluefunction, RedName,BlueName,Title):
    width = 12
    height = 10
    plt.figure(figsize=(width,height))

    ax1 = sns.kdeplot(Redfunction, color="r",label = RedName)
    ax2 = sns.kdeplot(Bluefunction, color="b",label = BlueName, ax=ax1 )
    plt.title(Title)
    plt.xlabel("Price(in dollars)")
    plt.ylabel('Proportion of Cars')
    plt.show()
    plt.close()

def PollyPlot(xtrain,xtest,y_train,y_test,lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width,height))

    xmax = max([xtrain.values.max(),xtest.values.max()])
    xmin = min([xtrain.value.min(),xtest.values.min()])

    x = np.arange(xmin,xmax,0.1)

    plt.plot(xtrain, y_train,"ro",label="Training Data")
    plt.plot(xtest.y_test,"go", label="Test Data")
    plt.plot(x,lr.predict(poly_transform.fit_transform(x.reshape(-1,1))),label="Predicted Function")
    plt.ylim(-10000,60000)
    plt.ylabel("Price")
    plt.legend()

# Parte 1 Entrenamiento y prueba (Training and testing)

y_data = df['price']

# arroja la información de precio en el dataframe x_data:

x_data = df.drop('price', axis=1)

# Ahora de forma al azar dividimos la información en información de entrenamiento y prueba, usando la función train_test_split.
# from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,test_size=0.10, random_state=1)

print("\nNumero de muestras de prueba: ",x_test.shape[0])
print("Numero de muestras de entrenamiento: ",x_train.shape[0],"\n")

# El parámetro test_size prepara la proporción de la información que se divide entre la preparación de las pruebas. Arriba, las pruebas son el 10% del total del dataset.

#------------------------------Pregunta 1
# Usa la función "train_test_split" para dividir el dataset para que el 40% de las muestras sean utilizadas para pruebas, prepara los parámetros "random_state" igual a zero. la salida de la función debe ser la siguiente: "x_train1","x_test1", "y_train1","y_test1".

print("Pregunta 1\nUsa la función \"train_test_split\" para dividir el dataset para que el 40% de las muestras sean utilizadas para pruebas, prepara los parámetro \"random_state\" igual a zero. la salida de la función debe ser la siguiente: \"x_train1\",\"x_test1\", \"y_train1\",\"y_test1\".\n")
x_train1,x_test1,y_train1,y_test1 = train_test_split(x_data,y_data,test_size=0.40,random_state=0)

print("Numero de muestras de prueba: ", x_test1.shape[0])
print("Numero de muestras de entrenamiento:",x_train1.shape[0],"\n")