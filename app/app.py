from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16  # Importar VGG16 aquí
import numpy as np
import os
import pickle

app = Flask(__name__)

# Cargar el modelo
model_cnn = load_model('./cnn.h5')
model_vgg16 = load_model('./vgg16.h5')

with open('class_mapping.pkl', 'rb') as f:
        class_mapping = pickle.load(f)

inverted_class_mapping = {v: k for k, v in class_mapping.items()}

# Cargar el mapeo de clases
with open('class_mapping.pkl', 'rb') as f:
    class_mapping = pickle.load(f)
    class_mapping = {v: k for k, v in class_mapping.items()}

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
            else:
                predicted_class = "Modelo no seleccionado correctamente."

    return render_template('index.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
