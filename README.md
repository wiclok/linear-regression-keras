# Regresión Lineal con Keras

Este proyecto demuestra un modelo de regresión lineal simple implementado utilizando Keras en Python. El objetivo de este proyecto es predecir el peso de una persona en función de su altura.

## Tabla de Contenidos

- [Introducción](#introducción)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Resultados](#resultados)

## Introducción

Este proyecto es un ejercicio práctico para aprender a utilizar Keras en tareas de regresión lineal. Entrenamos un modelo para predecir el peso a partir de datos de altura, visualizamos el proceso de entrenamiento y realizamos predicciones para nuevos datos.

## Requisitos

- Python 3.7+
- pandas
- numpy
- matplotlib
- tensorflow
- keras

## Instalación

1. Clona este repositorio:

   ```bash
   git clone https://github.com/wiclok/linear-regression-keras.git

2. Navega al directorio del proyecto:

    ```bash
    cd linear-regression-keras

3. Crea y activa un entorno virtual

    ```bash
    python -m venv my-env
    source my-env/Scripts/activate  # En Windows
    source my-env/bin/activate  # En MacOS/Linux

4. Intala las dependencias:
    ```bash
    pip install -r requirements.txt

## Uso

1. Asegúrate de tener el archivo altura_peso.csv en el directorio del proyecto.

2. Ejecuta el script principal:

    ```bash 
    python main.py

3. El programa entrenará un modelo de regresión lineal, mostrará gráficos de los resultados y realizará una predicción basada en una altura específica.

## Estructura del proyecto

- `main.py`: Script principal que contiene la lógica de entrenamiento, visualización y predicción del modelo.
- `altura_peso.csv`: Conjunto de datos utilizado para entrenar el modelo.

## Resultados

El modelo predice el peso de una persona en función de su altura. Los resultados se visualizan mediante gráficos que muestran la evolución del error cuadrático medio (ECM) durante el entrenamiento y la línea de regresión superpuesta sobre los datos originales.