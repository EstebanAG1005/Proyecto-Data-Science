import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
from autogluon.tabular import TabularPredictor

def preparar_imagen_para_modelo_tabular(ruta_imagen, modelo, target_size=(224, 224, 3)):
    # Cargar y preprocesar la imagen
    img = load_img(ruta_imagen, target_size=target_size)
    img = img_to_array(img) / 255

    # Aplanar la imagen
    img_flattened = img.reshape(1, -1)

    # Crear un DataFrame
    df = pd.DataFrame(img_flattened)

    # Hacer la predicción
    pred = modelo.predict(df)
    return pred

# Cargar el modelo
modelo1 = TabularPredictor.load("AutogluonModels/ag-20231116_225431", require_py_version_match=False)

# Ruta al directorio con imágenes
ruta_directorio = '../Proyecto-Data-Science-Tem/dataset/throwin/'

# Iterar sobre todas las imágenes en el directorio
for filename in os.listdir(ruta_directorio):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Asegurar que es una imagen
        ruta_completa = os.path.join(ruta_directorio, filename)
        prediccion = preparar_imagen_para_modelo_tabular(ruta_completa, modelo1)
        print(f"Imagen: {filename} - Predicción: {prediccion}")
