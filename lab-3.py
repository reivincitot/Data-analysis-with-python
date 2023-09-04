import pandas as pd
import numpy as np

print("para descargar el archivo que usaremos en este lab puedes usar este link: \"https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv\"\n")

print("Cargaremos la información y la guardaremos en el dataframe df:\n")

print("filename= \"D:/Curso ibm/Data analysis with python/automobileEDA.csv\"\n")

print("df = pd.read_csv(filename,header=0)\n")

print("print(df.head())")

filename = "D:/Curso ibm/Data analysis with python/automobileEDA.csv"

df = pd.read_csv(filename, header=0)
print(df.head())
