import pandas as pdb
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.model_selection import cross_val_score
from six.moves import cPickle as pickle

transformacion=make_column_transformer(
        (['Age','Fare'],StandardScaler()),#se escalan los números enteros
        (['Sex','Pclass','Embarked'],OneHotEncoder())#se transformar las variables categoritas en númericas, se crean columnas cuantos valores existan
    )
def cargarDatosParaEntrenamiento(file):
    return pdb.read_csv(file)


def elegirVariablesFuncionales(dataset,columns):
    columnas_funcionales=dataset.drop(columns=columns)
    return columnas_funcionales

def SepararVariables(dataset):
    print(dataset.isnull().sum())
    columna_Prediccion=dataset['Survived']
    columnas_entrenamiento=dataset.drop(columns=['Survived'])
    return columnas_entrenamiento,columna_Prediccion

def limpiarDatos(dataset):
    datos = dataset.dropna(subset=['Pclass', 'Sex', 'Age', 'Embarked', 'Fare'])
    return datos

def TransformarColumnas(dataset):
    columnasTransformadas=transformacion.fit_transform(dataset)
    return columnasTransformadas

def guardarData(dataset,prediccion):
    with open('titanic.pickle','wb') as f:
        pickle.dump([dataset,prediccion], \
                  f, pickle.HIGHEST_PROTOCOL)

def leerData():
    with open('titanic.pickle', "rb") as f:
      dataset,prediccion = pickle.load(f)
    return dataset,prediccion