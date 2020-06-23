import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.neural_network import MLPRegressor

#Importamos base de datos
datos = pd.read_csv("DatosMuertes.csv")
x = datos["Dias"]
y = datos["Muertes"]
X = x[:,np.newaxis]
i=0
mayor=0.0
menor=0.0
media=0.0

while True:

  
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X,y)

    mlr = MLPRegressor(solver='lbfgs',alpha = 1e-5, hidden_layer_sizes=(3,3),random_state=1)
    mlr.fit(X_train,y_train) # Entrenamos
    print(mlr.score(X_train,y_train))
    
    #Porcentaje de aceptacion
    if mlr.score(X_train,y_train) < 0.976:
        break
    #Predicciones en 1 dia 
    print("Predecir ", mlr.predict([[1]]))
    
    #Sacamos valor mas alto obtenido
    if mlr.predict([[1]]) > mayor:
        mayor=mlr.predict([[1]])

    #Sacamos valor mas bajo obtenido
    if mlr.predict([[1]]) < mayor:
        menor=mlr.predict([[1]])
    
    #Sacamos valor medio
    media=media+(mlr.predict([[1]]))
    i=i+1

media=media/i

print('\n'+"Valor mas alto obtenido: ", int(mayor)," muertes")
print("Valor medio: ", int(media)," muertes")
print("Valor mas bajo obtenido: ", int(menor)," muertes")

#Dibujar grafica 
datos=pd.read_csv('DatosMuertes.csv')
nuevo=datos[["Dias","Muertes"]]
grafica=sns.pairplot(nuevo,palette="Spectral")
