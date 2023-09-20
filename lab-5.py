# Después de completar este lab usted deberá ser capas de evaluar y refinar un modelo de predicción.

# Tabla de contenido:
# Evaluación del modelo
# Over-fitting, Under-fitting y Selección de Modelo
# Ridge Regression
# Grid Regression

import pandas as pd
import numpy as np
import seaborn as sns

path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/module_5_auto.csv'
df = pd.read_csv(path)