# Después de completar este lab usted deberá ser capas de evaluar y refinar un modelo de predicción.

# Tabla de contenido:
# Evaluación del modelo
# Over-fitting, Under-fitting y Selección de Modelo
# Ridge Regression
# Grid Regression
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ipywidgets import interact, interactive,fixed,interact_manual
from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/module_5_auto.csv'
df = pd.read_csv(path)
df.to_csv("module_5_auto.csv")

# Primero solo usaremos información numérica

df = df._get_numeric_data()
df = df.replace([np.inf, -np.inf], np.nan)
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

def PollyPlot(xtrain, xtest, y_train, y_test, model, poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    xmax = max([xtrain.values.max(), xtest.values.max()])
    xmin = min([xtrain.values.min(), xtest.values.min()])

    x = np.arange(xmin, xmax, 0.1)

    plt.plot(xtrain, y_train, "ro", label="Training Data")
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, model.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label="Predicted Function")
    plt.ylim(-10000, 60000)
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
x_train1,x_test1,y_train1,y_test1 = train_test_split(x_data, y_data,test_size=0.4 , random_state=0)

print("Numero de muestras de prueba: ", x_test1.shape[0])
print("Numero de muestras de entrenamiento:",x_train1.shape[0],"\n")

# Vamos a importar desde el modulo linear_regression LinearRegression
# From sklearn.linear_model import LinearRegression
# Creamos el objeto Linear Regression
lre=LinearRegression()

# Ajustamos el modelo usando la característica "horsepower":
lre.fit(x_train[['horsepower']],y_train)

# Vamos a calcular el R^2 en la información de prueba:

x = lre.score(x_test[['horsepower']],y_test)
print(x,"\n")
# Podemos ver que el R^2 es mas pequeño usando la información de prueba comparándola con la información de entrenamiento

y = lre.score(x_train[['horsepower']],y_train)
print(y,"\n")

#-----------------------Pregunta 2
# Encuentra el R^2 en el dataset usando el 40% del data set para pruebas
x_train2,x_test2,y_train2,y_test2 = train_test_split(x_data,y_data, test_size=0.4, random_state=0)

lre.fit(x_train2[['horsepower']],y_train2)
y = lre.score(x_test2[['horsepower']],y_test2)
print(y,"\n")

# A veces no tienes la suficiente información para realizar pruebas, Querrás realizar una validación cruzada (Cross Validation). Vamos a ver varios métodos que se pueden usar para la validación cruzada 

# Cross-Validation Score
# Vamos a importar el módulo cross_val_score desde el modulo model_selection
# form sklearn.model_selection import cross_val_score
# Ingresamos el objeto, característica ("horsepower"), y la información objetivo(y_data). el parámetro "cv" determina el numero de tomos o, en este caso son 4

Rcross = cross_val_score(lre,x_data[['horsepower']],y_data, cv=4)

print(Rcross,"\n")

print("La media de las muestras son: ",Rcross.mean(),"y la desviación estándar es: ", Rcross.std(),"\n")

# Podemos usar negative squared error como score definiendo el parámetro "scoring" y metric a "neg_mean_squared_error"

x = -1* cross_val_score(lre,x_data[['horsepower']],y_data,cv=4,scoring='neg_mean_squared_error')
print(x,"\n")

#-----------------------------------------------------Pregunta 3
#Calcula el promedio de R^2 usando 2 tomos, luego encuentra el promedio de R^2 para el segundo tomo usando "horsepower" como característica

Rcross2 = cross_val_score(lre,x_data[['horsepower']],y_data,cv=2)
print("La media de R^2 usando horsepower como característica es: ",Rcross2.mean())

# También puedes usar la función "cross_val_predict", para predecir la salida, la función divide la información en un numero especificado de tomos, con un tomo designado para pruebas, y los otros usados para el entrenamiento. Primero importamos la función:
# from sklearn.model_selection import cross_val_predict
# Ingresamos el objeto, la característica "horsepower", y la información objetivo y_data, el parámetro "cv" determina el numero de tomos, en este caso 4 . Podemos producir una salida
yhat = cross_val_predict(lre, x_data[['horsepower']],y_data,cv=4)
print(yhat[0:5])

##########################################-Parte dos, Overfitting, Underfitting y Model selection

# Resulta que la información de prueba, algunas veces se menciona como "out sample data", es una mucho mejor medida de cuan bien lo esta haciendo nuestro modelo en la vida real uan de estas razones es el overfitting
# Creemos el objeto Multiple linear regression y entrenemos el modelo usando "horsepower","curb-weight", "engine-size", y "highway-mpg" como caracteristicas

lr = LinearRegression()
x = lr.fit(x_train[['horsepower','curb-weight','engine-size','highway-mpg']],y_train)
print(x,"\n")

# Predicción usando la información de entrenamiento:

yhat_train = lr.predict(x_train[['horsepower','curb-weight','engine-size','highway-mpg']])
print(yhat_train[0:5],"\n")

# Predicción usando la información de prueba:
yhat_test = lr.predict(x_test[['horsepower','curb-weight','engine-size','highway-mpg']])
print(yhat_test[0:5],"\n")

# Vamos a realizar las evaluaciones del modelo usando nuestra información de entrenamiento y pruebas por separado. Primero importamos seaborn y matplotlib
# Vamos a examinar la distribución de los valores predecidos de la información de entrenamiento
Title = "Distribution plot de el valor predecido usando La información de entrenamiento VS La información de entrenamiento distribuida "
DistributionPlot(y_train,yhat_train,"Valor actual (entrenamiento)","Valores predecidos (entrenamiento)",Title)

# Figura 1 el gráfico de los valores predecidos usando la información de entrenamiento comparado con la información actual de los valores de entrenamiento
# hasta el momento, el modelo parece estar haciéndolo bien al aprender desde el entrenamiento del dataset. pero que pasa cuando el modelo encuentra nueva información para poner a prueba el data set. Cuando el modelo genera nuevos valores desde el dataset de prueba, podemos ver la distribución de los valores predecidos es muy diferente de los valores objetivo actuales

Title= "Gráfico de distribución del valor predecido usando información de prueba vs Información de distribución de la información de prueba"
DistributionPlot(y_test,yhat_test,"Valores actuales(Test)","Valores predecidos(Test)",Title)

# Figure 2: gráfico del valor predecido usando la información de prueba comparada a los valores actuales de prueba
# Comparando la figura 1 y la figura 2, es evidente que la distribución de la información de prueba en la figura 1 la información esta mejor ajustada, la diferencia en la figura 2,es aparentemente en el rango de 5000 a 15000. esto es donde la forma de la distribución es extremadamente diferente. veamos si la regresión polynomial también muestra una caída en la precision de la caída
# from sklearn.preprocessing import PolynomialFeatures

# Overfitting: El sobreajuste ocurre cuando el modelo se ajusta al ruido, pero no al proceso subyacente. Por lo tanto, al probar tu modelo utilizando el conjunto de pruebas, tu modelo no funciona tan bien, ya que está modelando el ruido, no el proceso subyacente que generó la relación. Creemos un modelo polinómico de grado 5.
# Vamos a usar el 55% de la información para entrenamiento y el resto para pruebas

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.55, random_state=0)

# Realizaremos una transformación polynomial de 5 grado en la característica "horsepower"

pr = PolynomialFeatures(degree=5)
x_train_pr = pr.fit_transform(x_train[['horsepower']])
x_test_pr = pr.fit_transform(x_test[['horsepower']])
pr

# Ahora vamos a crear un modelo Regresión lineal "poly" y entrenarlo
poly = LinearRegression()
poly.fit(x_train_pr, y_train)
# Podemos ver la salida de nuestro modelo usando el método "predict". Asignamos los valores a "yhat."

yhat = poly.predict(x_test_pr)
print(yhat[0:5])

# Tomaremos los primeros 5 valores y los compararemos con los valores objetivos actuales

print("Predicted values:", yhat[0:4])
print("True values:", y_test[0:4].values)
# Usaremos la función "Pollyplot" que definimos al comienzo del lab para mostrar la información de entrenamiento, información de prueba, y la función predecida

#PollyPlot(x_train['horsepower'], x_test['horsepower'], y_train, y_test, poly, pr)

# Figura 3: Un modelo de regresión polynomial donde los puntos rojos representan la información de entrenamiento, los puntos verdes representan la información de prueba, y la linea azul representa la predicción del modelo. Podemos ver que la función estimada aparece para seguir la información al rededor de los 200 horsepower, la función comienza a diverger de los puntos de información

# R^2 de la información de entrenamiento:
print(poly.score(x_train_pr,y_train))

# R^2 del data set

print(poly.score(x_test_pr,y_test))

# Vemos que el R^2 para la información de entrenamiento es 0.5567 mientras que el R^2 de la información de prueba fue -29.87. Entre mas bajo el R^2 peor el modelo. Un R^2 negativo es un signo de overfitting

# Veamos como el R^2 cambia la información de prueba para diferentes orden polynomial y el grafico como resultado

Rsqu_test = []

order = [1,2,3,4]
for n in order:
    pr = PolynomialFeatures(degree=n)
    #-----
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    #-----
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    #-----
    lr.fit(x_train_pr,y_train)
    #----
    Rsqu_test.append(lr.score(x_test_pr,y_test))

plt.plot(order,Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Usando información de Prueba')
plt.text(3,0.75,'R^2 Maximo')