#https://medium.com/@jcrispis56/una-introducci%C3%B3n-completa-a-redes-neuronales-con-python-y-tensorflow-2-0-b7f20bcfebc5
#tensorflow=https://blog.mimacom.com/getting-started-tensorflow-spring/
#keras=https://towardsdatascience.com/deploying-keras-deep-learning-models-with-java-62d80464f34a
import pandas as pdb
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.model_selection import cross_val_score
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from flask import Flask, request, jsonify
from tensorflow import keras
import cleanData
import json
from pandas.io.json import json_normalize


columns=['PassengerId','Name','SibSp','Parch','Ticket','Cabin']
df,label=cleanData.leerData()


def predecir(modelo,predictObject):
    cnames = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
    data = [[predictObject.Pclass, predictObject.Sex,predictObject.Age, predictObject.Fare,predictObject.Embarked]]
    predictDataset = pdb.DataFrame(data=data, columns=cnames)
    predictDataset = cleanData.transformacion.transform(predictDataset)
    return modelo.predict_classes(predictDataset)


# Your API definition
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    #lr = joblib.load("model.pkl") # Load "model.pkl"
    json_ = request.json

    #entrenamiento_procesado=cleanData.TransformarColumnas(df)
    
    modelo_entrenado=keras.models.load_model('titanic.h5')
    entrenamiento_procesado=cleanData.TransformarColumnas(df)
    
    class Survived:
        Pclass=1
        Sex='male'
        Age=60
        Fare='0'
        Embarked='C'
    surv = Survived()
    class Survived:
        Pclass=json_['Pclass']
        Sex=json_['Sex']
        Age=json_['Age']
        Fare=json_['Fare']
        Embarked=json_['Embarked']
    surv = Survived()
    print(predecir(modelo_entrenado,surv)[0][0])
    return str(predecir(modelo_entrenado,surv)[0][0])



if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345
    app.run(port=port, debug=True)