# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 20:14:48 2024

@author: KGP
"""

# Regresión lineal múltiple

# Cómo importar las librerías
import numpy as np # contiene las herrarmientas matemáticas para hacer los algoritmos de machine learning
import matplotlib.pyplot as plt #pyplot es la sublibrería enfocada a los gráficos, dibujos
import pandas as pd #librería para la carga de datos, manipular, etc

# Importar el dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
# iloc sirve para localizar por posición las variables, en este caso independientes
# hemos indicado entre los cochetes, coge todas las filas [:(todas las filas), :-1(todas las columnas excepto la última]
# .values significa que quiero sacar solo los valores del dataframe no las posiciones

# Codificar datos categóricos

x = pd.get_dummies(dataset)
x


# Evitar la trampa de la variable ficticia
x.drop(['Profit', 'State_California'], axis = 1, inplace=True)

# Reordeno columnas para que queden los estados al principio

x = x[["State_New York", "State_Florida", "R&D Spend", "Administration", "Marketing Spend"]]

# Dividir el dataset en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0) # random_state podría coger cualquier número, es el número para poder reproducir el algoritmo

# Escalado de variables. Siguiente código COMENTADO porque se usa mucho pero no siempre
"""from sklearn.preprocessing import StandardScaler # Utilizarlo para saber que valores debe escalar apropiadamente y luego hacer el cambio
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test) #hacemos transform sin "fit" para que haga la transformación con los datos del transform de entrenamiento"""

# Ajustar el modelo de regresión lineal múltiple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train, y_train)

# Predicción de los resultados en el conjunto de testing
y_pred = regression.predict(x_test)

# Eliminación (eliminicación hacia atrás) de las variables que no son estadísticamente significativas del modelo para mejor predicción del modelo
# hemos de detectar cuales de los coeficientes de las variables ind tienen un valor muy cercano a 0 a través del p > 0.05 para ser eliminados
import statsmodels.api as sm # esta librería ayuda a añadir/quitar variables
x = np.append(arr = np.ones((50, 1)).astype(int), values = x, axis = 1) # Añadimos una columna de 1 para representar el término independiente (coorodenada en el origen) y saber si lo que sobra, ya que este no sale reflejado en la tabla y veremos luego su valor p
SL = 0.05
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
regression_OLS = sm.OLS(endog=y, exog=x_opt).fit() # Necesitamos un nuevo regresor para reajustar el modelo con la nueva estructura de datos en la que hemos añadido la columna de 1. sm.OLS Librería que vamos a utilizar para la eliminación hacia atrás
# endog que es la variable que queremos predecir y exog que representa la matriz de características 
regression_OLS.summary() # Nos devuelve toda la información de la OLS

#volvemos a pegar el código anterior con la variable con el valor P más elevado eliminada
x_opt = x[:, [0, 1, 3, 4, 5]]
regression_OLS = sm.OLS(endog=y, exog=x_opt).fit() 
regression_OLS.summary() 

#volvemos a pegar el código anterior con la variable con el valor P más elevado eliminada
x_opt = x[:, [0, 3, 4, 5]]
regression_OLS = sm.OLS(endog=y, exog=x_opt).fit() 
regression_OLS.summary() 

#volvemos a pegar el código anterior con la variable con el valor P más elevado eliminada
x_opt = x[:, [0, 3, 5]]
regression_OLS = sm.OLS(endog=y, exog=x_opt).fit() 
regression_OLS.summary() 

#volvemos a pegar el código anterior con la variable con el valor P más elevado eliminada
x_opt = x[:, [0, 3]]
regression_OLS = sm.OLS(endog=y, exog=x_opt).fit() 
regression_OLS.summary() # Ha quedado un modelo de regresión lineal simple. Hay otras mejores formas de criterio para eliminar, ya que en este caso nos hemos cargado un p = 0.06 que estaba al filo y pudiera ser bueno teniendo en cuenta otros datos de la tabla summary.



