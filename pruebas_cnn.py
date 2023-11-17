from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16  # Importar VGG16 aquí
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
from autogluon.tabular import TabularPredictor
import numpy as np
import os
import pickle
import time

model_cnn = load_model('./app./cnn.h5')

with open('./app/class_mapping.pkl', 'rb') as f:
        class_mapping = pickle.load(f)

inverted_class_mapping = {v: k for k, v in class_mapping.items()}

# Cargar el mapeo de clases
with open('./app/class_mapping.pkl', 'rb') as f:
    class_mapping = pickle.load(f)
    class_mapping = {v: k for k, v in class_mapping.items()}


def predict_class_cnn(image_path):
    start_time = time.time()
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    prediction = model_cnn.predict(img)
    # Aquí puedes procesar la salida de 'prediction' según sea necesario
    result = np.argmax(prediction, axis=1)
    predicted_class_name = inverted_class_mapping[result[0]]
    end_time = time.time()
    prediction_time = end_time - start_time
    model="CNN"
    return predicted_class_name, prediction_time, model  # Just return the result


classes = ["play", "challenge", "throwin"]

for _class in classes:

    # Ruta al directorio con imágenes
    ruta_directorio = f'../Proyecto-Data-Science-Tem/dataset/{_class}/'

    lines = []

    # Iterar sobre todas las imágenes en el directorio
    for filename in os.listdir(ruta_directorio):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Asegurar que es una imagen
            ruta_completa = os.path.join(ruta_directorio, filename)
            predicted_class, prediction_time, model = predict_class_cnn(ruta_completa)
            lines.append((predicted_class, filename))

    # Convertir la lista de tuplas en un DataFrame
    df = pd.DataFrame(lines, columns=['predicted_class', 'filename'])

    # Escribir el DataFrame a un archivo Excel
    nombre_del_archivo = f'./cnn/{_class}.xlsx'
    df.to_excel(nombre_del_archivo, index=False)