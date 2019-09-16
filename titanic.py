#https://medium.com/@jcrispis56/una-introducci%C3%B3n-completa-a-redes-neuronales-con-python-y-tensorflow-2-0-b7f20bcfebc5
#tensorflow=https://blog.mimacom.com/getting-started-tensorflow-spring/
#keras=https://towardsdatascience.com/deploying-keras-deep-learning-models-with-java-62d80464f34a
import pandas as pdb
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import cleanData


df=pdb.read_csv('train.csv')


PCLASS = 'Pclass'
SEX = 'Sex'
AGE = 'Age'
EMBARKED = 'Embarked'
FARE = 'Fare'
columns=['PassengerId','Name','SibSp','Parch','Ticket','Cabin']

def modelo(train,train_predict):
    model = Sequential()
    model.add(Dense(32, input_shape=(train.shape[1],), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit(train, train_predict, epochs=100, batch_size=64)
    return model

def predecir(modelo,predictObject):
    cnames = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
    data = [[predictObject.Pclass, predictObject.Sex,predictObject.Age, predictObject.Fare,predictObject.Embarked]]
    predictDataset = pdb.DataFrame(data=data, columns=cnames)
    predictDataset = cleanData.transformacion.transform(predictDataset)
    return modelo.predict_classes(predictDataset)


def guardarModelo(modelo):
    modelo.save('titanic.h5')


#ExplorarDataset()
variables_funcionales=cleanData.elegirVariablesFuncionales(df,columns)

datos_limpios=cleanData.limpiarDatos(variables_funcionales)

entrenamiento,prediccion=cleanData.SepararVariables(datos_limpios)

cleanData.guardarData(entrenamiento,prediccion)

entrenamiento_procesado=cleanData.TransformarColumnas(entrenamiento)

modelo_entrenado=modelo(entrenamiento_procesado,prediccion)



class Survived:
  Pclass=1
  Sex='female'
  Age=60
  Fare=0
  Embarked='C'

surv = Survived()
print(predecir(modelo_entrenado,surv)[0][0])
guardarModelo(modelo_entrenado)
