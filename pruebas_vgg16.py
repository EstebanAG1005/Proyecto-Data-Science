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

model_vgg16 = load_model('./app./VGG16.h5')

with open('./app/class_mapping.pkl', 'rb') as f:
    class_mapping = pickle.load(f)
    class_mapping = {v: k for k, v in class_mapping.items()}
    inverted_class_mapping = {v: k for k, v in class_mapping.items()}

# Función para predecir la clase de la imagen
def predict_class_vgg16(image_path):
    start_time = time.time()
    # Cargar la imagen y preprocesarla
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Obtener características usando VGG16 (sin las capas superiores)
    base_model = VGG16(weights='imagenet', include_top=False)
    features = base_model.predict(img)

    # Aplanar las características para coincidir con el entrenamiento
    flattened_features = features.reshape(1, 7*7*512)

    # Normalizar los píxeles
    flattened_features = flattened_features / flattened_features.max()

    # Hacer la predicción
    prediction = model_vgg16.predict(flattened_features)
    result = np.argmax(prediction, axis=1)
    end_time = time.time()
    prediction_time = end_time - start_time
    model="VGG16"
    return class_mapping[result[0]], prediction_time, model

# Ruta al directorio con imágenes
ruta_directorio = '../Proyecto-Data-Science-Tem/dataset/play/'

# Iterar sobre todas las imágenes en el directorio
for filename in os.listdir(ruta_directorio):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Asegurar que es una imagen
        ruta_completa = os.path.join(ruta_directorio, filename)
        predicted_class, prediction_time, model = predict_class_vgg16(ruta_completa)
        print(f"Imagen: {filename} - Predicción: {predicted_class}")
