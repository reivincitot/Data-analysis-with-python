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
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score

filename = "D:/Curso ibm/Data analysis with python/automobileEDA.csv"
df = pd.read_csv(filename,header=0)
print(df.head(0),"\n")

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
print(lm,"\n")

# Como puede "highway-mpg" ayudarnos a predecir el precio de car, para este ejemplo, queremos mirar como highway-mpg nos puede ayudar a predecir el precio del auto. Usando Regresión lineal simple (Simple linear regression), podemos crear una función lineal con "highway-mpg" como variable predictora y "price" como variable de respuesta
X = df[['highway-mpg']]
Y = df[['price']]

# Llena el modelo lineal usando "highway-mpg":
lm.fit(X,Y)

# Obtendremos una predicción como salida:

Yhat= lm.predict(X)
print(Yhat[0:5],"\n")

# Cual es el valor de la intersección (a)?

print(lm.intercept_,"\n")

# Cual es el valor del pendiente (b)

print(lm.coef_,"\n")

# Cual es el modelo lineal estimado final que obtenemos?. Como vemos arriba deberíamos obtener un modelo lineal final con la estructura Yhat= a+bX

# Complementando en los valores actuales que obtenemos: Price = 33842.31 - 821.73 x highway-mpg

# Pregunta 1 a): Crea un objeto Regresión lineal llamado "lm1"

lm1= LinearRegression()
print(lm1,"\n")

# Pregunta 1 b): Entrena al modelo usando "engine-size" como la variable independiente y "price" como la variable dependiente

X = df[["engine-size"]]
Y = df[["price"]]

print(lm1.fit(X,Y),"\n")

# Pregunta 1c): Encuentra el pendiente y intersección del modelo

print(lm1.coef_,"\n")

# intersección

print(lm1.intercept_,"\n")

#Pregunta 1 d): cual es la ecuación de la linea predecída puedes usar Yhat o "engine-size" o "price"

Yhat = -7963.34+166.86*X
print(Yhat[0:5],"\n")
Price = -7963.34+166.86*df[["engine-size"]]
print(Price,"\n")

# Regresión lineal multiple. Que sucede si queremos predecir el precio de un vehículo usando mas de una variable?

# si queremos usar mas de una variable en nuestro modelo para predecir el precio de un vehículo, podemos usar la Regresión lineal multiple (Multiple Linear Regression). Regresión lineal multiple es muy similar a la Regresión Lineal Simple, pero este método es usado para explicar la relación entre una respuesta constante, variable(dependiente o dependent) y dos o mas predictores de variables (independiente o independent). La mayoría de los modelos de regresión del mundo real implican multiples predictores. Ilustraremos la estructura usando cuatro variables predictoras, pero el resultado puede generalizar a cualquier numero entero

# Y: Variable de respuesta
# X_1: Variable predictora 1
# X_2: Variable predictora 2
# X_3: Variable predictora 3
# X_4: Variable predictora 3

#       a: intercepta
# b_1: Coeficiente de variable 1
# b_2: Coeficiente de variable 2
# b_3: Coeficiente de variable 3
# b_4: Coeficiente de variable 4

# La ecuación es dada por 

# Yhat = a + b_1X_1 + b_2X2 + b_3X_3+b_4X_4
# Desde la sección previa sabemos que otros buenos predictores pueden ser:

# Horsepower
# Curb-weight
# Engine-size
# Highway-mpg

# Desarrollemos un modelo usando estas variables como variables predictoras

Z = df[['horsepower','curb-weight','engine-size','highway-mpg']]

# Llena el modelo lineal usando las cuatro variables mencionadas arriba.

lm.fit(Z,df['price'])

# Cual es el valor del interceptor(a)?

print("El valor del interceptor (a) es: ",lm.intercept_,"\n")

# Cuales son los valores de los coeficientes(b1,b2,b3,b4)?

print(lm.coef_,"\n")

# Cual es el estimado final del modelo lineal que obtenemos?

# Como vimos arriba, deberíamos obtener un modelo lineal con esta estructura Yhat = a + b_1X_1 + b_2X2 + b_3X_3+b_4X_4

# Cual es la función lineal que obtenemos en este ejemplo?

# Price = -15678,742628061467 + 52,65851272 x horsepower + 4,69878948 x curb-weight + 81,95906216 x  engine-size + 33,58258185 x highway-mpg

# Pregunta 2 a): Crea y entrena un Modelo regresión lineal multiple "lm2" donde la variable de respuesta es "price", y la variable predictora es "normalized-losses" y "highway-mpg"
lm2 = LinearRegression()
Z= df[["normalized-losses","highway-mpg"]]
print(lm2.fit(Z,df['price']),"\n")

# Pregunta 2 b): Encuentra el coeficiente

print(lm2.coef_)

# 2 Evaluación del modelo usando visualización, ahora que hemos desarrollado algunos modelos, como evaluamos nuestros modelos y decidimos cual es el mejor? una manera de hacerlo es usando un visualizador.

# Regression plot. Cuando se trata de una regresión lineal simple, un ejemplo de visualizar el llenado de nuestro modelo es usando regression plots.
# Este plot mostrara una combinación de puntos de información dispersos (scatterplots), también como la regresión lineal pasando a traves de la información. Esto nos da una estimación razonable de la relación entre las dos variables, la fuerza de la correlación, asi como la dirección(correlación positiva o negativa)

# Vamos a visualizar highway-mpg como un potencial predictor de la variable price:

width = 12
height = 10

plt.figure(figsize=(width,height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)
plt.show()

# Podemos ver desde este entramado que el precio tiene una correlación negativa con respecto a highway-mpg dado que la pendiente de regresión es negativa. Una cosa para mantener en mente cuando miramos el entramado de regresión es mirar cuan dispersos se encuentran los puntos de entrada con respecto a la linea de regresión, esto te dará una buena indicación de la varianza de la información y si un modelo sera el mas adecuado o no. si la información esta demasiado alejada de la linea de regresión, este modelo puede no ser el mejor para esta información

# Vamos a comparar esto entramado con el entramado de "peak-rpm"

plt.figure(figsize=(width,height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)
plt.show()

# Pregunta 3 Dado el entramado de regresión de arriba, es "peak-rpm" o "highway-mpg" mas fuertemente correlacionado con "price"?. usa el método .corr() para verificar la respuesta

df[["peak-rpm","highway-mpg","price"]].corr()

# Residual plot una buena manera de visualizar la variación de la información es usar un residual plot
# Que es residual? las diferencias entre el valor observado (Y) y el valor predecido (Yhat) es llamado residual(e), cuando miramos al regression plot, el residuo es la distancia desde el punto de entrada y la linea de regresión

# Así que ¿que es un residual plot? Un residual plot es un gráfico de residuos en el eje vertical Y la variable independiente en el eje horizontal X
# A que le prestamos atención al mirar un residual plot. Miramos a la dispersion de los residuos

# si el punto en un plot residual están dispersos al azar al rededor del eje X entonces el modelo linear es el apropiado par esta información
# Por que es eso? Esparcidos al azar significa que la varianza es constante, y por lo tanto el modelo lineal esta bien ajustado para esta información

width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(x="highway-mpg",y="price",data=df)
plt.show()

# Que nos muestra esta trama podemos ver que los datos de entrada no están dispersos de forma aleatoria al rededor del eje x, llevándonos a pensar, que un modelo no lineal seria mas apropiado para esta información

# Multiple linear regression

# Como visualizamos un modelo para una regresión lineal multiple? esto se vuelve un poco mas complicado por que no puedes visualizar con regresión o residual plot

# Una manera de hacerlo es mirar el ajuste del modelo mirando el entramado de distribución y comparando con la distribución de los valores verdaderos

# Primero haremos una predicción
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]

Y_hat=lm.predict(Z)

plt.figure(figsize=(width,height))

ax1= sns.kdeplot(df['price'],color="r",label="Actual value", fill=False)
sns.kdeplot(Y_hat, color="b", label="Fitted values", ax= ax1, fill=False)

plt.title("Actual values vs Fitted values for price")
plt.xlabel("Price (in dollars)")
plt.ylabel("Proportion of Cars")

plt.show()
plt.close

# Regresión polinómica (Polynomial Regression) es un caso particular de la generalidad de la regresión lineal o regresión lineal multiple
# Obtenemos relaciones no lineales al elevar al cuadrado o establecer términos de orden superior de las variables predictoras.
# hay diferentes ordenes en la regresión de polinomios:
# Cuadrática - segundo orden: Yhat = a + b1X + b2X**2
# Cubico - tercer orden: Yhat = a + b1X +b2X**2 + b3X**3
# Alto orden: Y= a+b1x**2 + b2x**2 + b3**3

# Vimos anteriormente que un modelo lineal no provee el resultado mas ajustado mientras se usa "highway-mpg" como variable predictora. Veremos si podemos ajustar el modelo de polinomios a la información

# Usaremos la siguiente función:

def PlotPolly(model,independent_variable, dependent_variable, Name):
    x_new = np.linspace(15,55,100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variable,'.',x_new,y_new,'-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898,0.898,0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()

# Vamos a tomar las variables
x = df['highway-mpg']
y = df['price']

# Vamos a ajustar el polinomio usando la función polyfit, después usa la función poly1d para:

f = np.polyfit(x,y,3)
p = np.poly1d(f)
print(p,"\n")

# Vamos a entramar la función
PlotPolly(p,x,y,'highway-mpg')

# Pregunta 4 crea 11 modelos polynomial ordenados con la variable "x" y "y" de arriba.

f1 = np.polyfit(x,y,11)
p1 = np.poly1d(f1)
print(p1,"\n")
PlotPolly(p1,x,y,'Highway MPG')

# La expresión analítica para una función polinómica multivariada se vuelve complicada. Por ejemplo, la expresión para un polinomio de segundo orden (grado=2) con dos variables se da por: Yhat = a + b_1X_1 + b_2X_2 + b_3X_1X_2 + b_4X_1**2 + b_5X_2**2

# Podemos crear un objeto de caracteristicas polinomiales de grado 2:

pr= PolynomialFeatures(degree=2)
print(pr,"\n")

Z_pr = pr.fit_transform(Z)

print(Z.shape,"\n")

print(Z_pr.shape,"\n")

# Pipeline o secuencia de pasos: data pipelines simplifican los pasos del procesado de la información. Nosotros usamos el modulo Pipeline para crear un pipeline. También usamos StandardScaler como un paso en nuestra pipeline from sklearn.pipeline import Pipeline from sklearn.preprocessing import StandardScaler

# Creamos el pipeline comenzando por crear una lista de tuplas incluyendo los nombres del modelo o el estimador y su correspondiente constructor

Input = [('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]

# Ingresamos la lista como un argumento al constructor del pipeline

pipe = Pipeline(Input)
print(pipe,"\n")

# Primero, convertimos la información de tipo Z a tipo float para evitar las alertas de conversión que pueden aparecer como resultados de StandardScaler tomando float inputs. Entonces , podemos normalizar la información, para realizar una transformación y ajuste del modelo en simultaneo.

Z = Z.astype(float)
print(pipe.fit(Z,y),"\n")

# Similarmente, podemos normalizar la información, realizaremos una transformación y una predicción en simultaneo

ypipe = pipe.predict(Z)
print(ypipe[0:4])

# Pregunta 5 Crea un pipeline que estandarice la información , entonces produce una predicción usando una regresión lineal usando la característica Z y el objetivo y.

Input = [('Scale',StandardScaler()),('model', LinearRegression())]

pipe = Pipeline(Input)

pipe.fit(Z,y)

ypipe = pipe.predict(Z)
print(ypipe[0:10])

# Media para una muestra de evaluación. Cuando evaluamos nuestro modelo, no solo queremos visualizar los resultados, también queremos una medida cuantitativa para determinar cuan certero es nuestro modelo. Dos importantes medidas que usualmente se usan en estadísticas para determinar la precisión del modelo:
# R**/R-squared
# Mean Squared Error (MSE)
# R-squared
# R squared, también conocido como determinación de coeficiente, es una medida que  indica cuan cerca esta la información de la linea de regresión
# El valor de R-squared es el porcentaje de la variación de la variable respuesta (response) (y)  eso es explicado por el modelo lineal

# Mean Squared Error (MSE)

# Mean squared error mide el promedio de los cuadrados de los errores. es la diferencia entre  el valor actual (y) y el valor estimado (ŷ).

# Modelo 1 Regresión lineal simple. Vamos a calcular el R^2:

lm.fit(X,Y)
print("El R-square es: ",lm.score(X,Y))

# Podemos decir que ~49,659% de la variación del precio es explicado por este simple modelo lineal, el cual hemos construido usando la información de highway-mpg.

# Calculemos el MSE:
# Podemos predecir la salida i.e, "yhat" usando el método predict, donde X es el input de la variable

Yhat=lm.predict(X)
print("La salida de los primeros cuatro valores predecidos es: ", Yhat[0:4])

# Importamos la función mean_square_error del modulo metrics(esta importada arriba junto con todas las importaciones)

mse = mean_squared_error(df['price'],Yhat)
print("La media de errores cuadrados de price y valor predictor es: ", mse)

# Modelo 2: Regresión Lineal Multiple:
# Vamos a calcular R^2

lm.fit(Z,df['price'])
r2=lm.score(Z,df['price'])
print("El valor de R^2 es: ", r2)

# Podemos decir que ~80,896% de la variación de precio es explicada por multiples regresiones lineales "multi_fit"
# Vamos a calcular el MSE:
# Producimos una predicción:

Y_predict_multifit = lm.predict(Z)

# Comparamos los valores predecidos con los resultados actuales:
m2e = mean_squared_error(df['price'], Y_predict_multifit) 
print("El promedio de errores cuadráticos de price y el valor predecido usando multifit es: ", m2e)

# Modelo 3 Ajuste polynomial. Vamos a calcular R^2, importaremos la función r2_score del modulo metrics ya que estamos usando una función diferente
# Aplicamos la función para obtener el valor de R^2
r_squared = r2_score(y,p(x))
print("El valor de R-square es: ", r_squared)

# Podemos decir que ~67,419%de la variación de precio es explicado por este ajuste polynomial (Polynomial fit)

# MSE
# También podemos calcular el valor de MSE
print(mean_squared_error(df['price'],p(x)))

# Predicción y toma de decisiones
# Predicción
# En la sección previa, entrenamos el modelo usando el método fit. Ahora usaremos el método predict para producir una predicción. importaremos pyplot, también usaremos algunas funciones de numpy

new_input = np.arange(1,100,1).reshape(-1,1)

# ajustando el modelo:
lm.fit(X,Y)
print(lm)

# Produciendo una predicción:
Yhat = lm.predict(new_input)
print(Yhat[0:5])

# Podemos entramar la información:

plt.plot(new_input,Yhat)
plt.show()

# Toma de decisiones: Determinar un buen modelo

# Ahora que hemos visualizado los diferentes modelos y generar los valores de R-squared y el MSE para ajustarlos, como determinamos un buen modelo?

# Que es un buen valor de R-squared?
# Cuando comparas modelos, el modelo con el mayor valor de R-squared es el que se ajusta mejor, para la información

# Que es un buen MES?
# Cuando comparas modelos, el modelo con el menor valor SME es el mas ajustado para la información

# Miremos los valores de diferentes modelos
# Simple Linear Regression: Usando highway-mpg como variable predictora de Price.

# R-squared: 0,49659118843391759
# MSE: 3,16 * 10^7

# Multiple Linear Regression: Usando horsepower, Curb-weight,Engine-size y Highway-mpg como variables predictoras de Price

# R-squared: 0,80896354913783497
# MSE: 1,2 X 10^7

# Polynomial Fit: Usando Highway-mpg como variable predictora de Price

# R-squared: 0,6741946663906514
# MSE: 2,05 X 10^7

# Simple Linear Regression model (SLR) vs Multiple Linear Regression model (MLR)
# Usualmente, entre mas variables tengas, mejor tu modelo es prediciendo, pero esto no es siempre cierto. A veces no tendrás suficiente información, te encontraras con problemas numéricos, o muchas de las variables no serán útiles o incluso actuaran como ruido. Como resultado tu siempre deberás revisar el MSE y el R^2

# Para comparar el resultado de los modelos MLR vs SLR, miramos a la combinación de los dos R-squared y MSE para tener la mejor conclusión acerca del ajuste del modelo

# MSE: El MSE de SLR es 3,16 x 10^7 mientras que MLR tiene un MSE DE 1,2,X 10^7 el MSE de MLR es mas pequeño
# R-squared: en este caso, podemos ver que hay una gran diferencia entre el R-squared de SLR y el de MLR, el R-squared de SLR es (~0,497), bastante pequeño comparado con el R-squared de MLR(~0,809)

# Este R-squared en combinación con el MSE  muestra que MLR en este caso es un mejor modelo comparado con SLR

# Simple Linea Model (SLR) vs Polynomial fit

# MSE: podemos ver que Polynomial fit echo abajo el MSE, dado que el MSE es pequeño comparado con el SLR
# R-squared El R-squared para Polynomial fit es mas grande que el R-squared de SLR, asi que Polynomial fit llevo hacia arriba el R-squared un poco

# Dado que Polynomial fit resulto en un bajo MSE y un alto R-squared, podemos concluir que esto fue mejor ajustado que un Simple Linear Regression para predecir "price" con "highway-mpg" como variable predictora

# Multiple Linea Regression (MLR) vs Polynomial Fit

# MSE: el MSE para MLR es pequeño que el MSE de Polynomial Fit
# R-squared: El R-squared para el MLR es también mucho mas grande que el de Polynomial fit

# Conclusión:
# Comparando estos tres modelos, podemos concluir que el modelo MLR es el mejor modelo, para poder predecir price de nuestro dataset. Estos resultados tienen sentido ya que tenemos 27 variables y sabemos que mas de una variable son potenciales predictores del precio final del car