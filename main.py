import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.api.layers import Dense
from keras.api.models import Sequential
from keras.api.optimizers import SGD

def read_file(filename):
    data = pd.read_csv(filename)
    x = data['Altura'].values
    y = data['Peso'].values
    return x, y

def create_model():
    model = Sequential()
    model.add(Dense(1, input_shape=(1,), activation="linear"))
    optimizer = SGD(learning_rate=0.00001)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def train_history(model, x, y):
    history = model.fit(x, y, epochs=20, batch_size=len(x), verbose=0)
    return history

def show_results(history, model, x, y):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.title('Error Cuadrático Medio vs. Número de Épocas')
    plt.xlabel('Épocas')
    plt.ylabel('Error Cuadrático Medio (ECM)')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Datos Originales')
    plt.plot(x, model.predict(x), color='red', label='Línea de Regresión')
    plt.title('Altura vs. Peso con Línea de Regresión')
    plt.xlabel('Altura')
    plt.ylabel('Peso')
    plt.legend()
    plt.show()

def predict_weight(model, height):
    weight = model.predict(np.array([height]))
    print(f'La predicción del peso para una persona con una altura de {height} cm es de {weight[0][0]:.2f} kg')

def run_program():
    filename = 'altura_peso.csv'
    x, y = read_file(filename)
    
    model = create_model()
    
    history = train_history(model, x, y)
    
    show_results(history, model, x, y)
    
    # Prediccion para una altura especifica
    predict_weight(model, 170)

if __name__ == '__main__':
    run_program()
