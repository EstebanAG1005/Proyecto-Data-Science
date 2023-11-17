from flask import Flask, request, render_template
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

app = Flask(__name__)

# Cargar el modelo
model_cnn = load_model('./cnn.h5')
model_vgg16 = load_model('./vgg16.h5')
modelo__autogluon = TabularPredictor.load("../AutogluonModels/ag-20231116_225431", require_py_version_match=False)

with open('class_mapping.pkl', 'rb') as f:
        class_mapping = pickle.load(f)

inverted_class_mapping = {v: k for k, v in class_mapping.items()}

# Cargar el mapeo de clases
with open('class_mapping.pkl', 'rb') as f:
    class_mapping = pickle.load(f)
    class_mapping = {v: k for k, v in class_mapping.items()}



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

# Función para predecir la clase de la imagen
def predict_class_vgg16(image_path):
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
    print("Resultado de la predicción:", result)
    print("Mapeo de clases:", class_mapping)

    return class_mapping[result[0]]

def predict_class_cnn(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    prediction = model_cnn.predict(img)
    # Aquí puedes procesar la salida de 'prediction' según sea necesario
    result = np.argmax(prediction, axis=1)
    predicted_class_name = inverted_class_mapping[result[0]]
    return predicted_class_name  # Just return the result

def predict_class_autogluon(image_path):
    prediccion = preparar_imagen_para_modelo_tabular(image_path, modelo__autogluon)
    # Convertir la predicción (que es una serie de Pandas) a un valor único o cadena
    prediccion_str = prediccion.iloc[0] if not prediccion.empty else "No prediction"
    return str(prediccion_str)


@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    predicted_class = None  # Initialize with None

    if request.method == 'POST':
        image_file = request.files['image']
        model_choice = request.form.get('model_choice')

        if image_file:
            image_path = os.path.join('images', image_file.filename)
            image_file.save(image_path)

            if model_choice == 'cnn':
                predicted_class = predict_class_cnn(image_path)
            elif model_choice == 'vgg16':
                predicted_class = predict_class_vgg16(image_path)
            elif model_choice =='autogluon':
                predicted_class = predict_class_autogluon(image_path)
            else:
                predicted_class = "Modelo no seleccionado correctamente."

    return render_template('index.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
