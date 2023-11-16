# from flask import Flask, request, render_template
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import numpy as np
# import pickle

# app = Flask(__name__)

# # Cargar el modelo
# model = load_model('./vgg16.h5')

# @app.route('/', methods=['GET', 'POST'])
# def upload_predict():

#     with open('class_mapping.pkl', 'rb') as f:
#         class_mapping = pickle.load(f)

#     # Invertir el mapeo para facilitar la búsqueda por índice
#     inverted_class_mapping = {v: k for k, v in class_mapping.items()}

        
#     if request.method == 'POST':
#         image_file = request.files['image']
#         if image_file:
#             image_location = "./images/" + image_file.filename
#             image_file.save(image_location)

#             img = load_img(image_location, target_size=(224, 224))
#             img = img_to_array(img)
#             img = np.expand_dims(img, axis=0)
#             img = img / 255.0

#             prediction = model.predict(img)
#             # Aquí puedes procesar la salida de 'prediction' según sea necesario
#             result = np.argmax(prediction, axis=1)
#             predicted_class_name = inverted_class_mapping[result[0]]
#             return render_template('index.html', prediction=predicted_class_name)
    
#     return render_template('./index.html')

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16  # Importar VGG16 aquí
import numpy as np
import os
import pickle

app = Flask(__name__)

# Cargar el modelo VGG16
model = load_model('../modelo_entrenado_VGG16.h5')

# Cargar el mapeo de clases
with open('class_mapping.pkl', 'rb') as f:
    class_mapping = pickle.load(f)
    class_mapping = {v: k for k, v in class_mapping.items()}

# Función para predecir la clase de la imagen
def predict_class(image_path):
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
    prediction = model.predict(flattened_features)
    result = np.argmax(prediction, axis=1)
    print("Resultado de la predicción:", result)
    print("Mapeo de clases:", class_mapping)

    return class_mapping[result[0]]
    
@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_path = os.path.join('images', image_file.filename)
            image_file.save(image_path)

            predicted_class = predict_class(image_path)
            return render_template('index.html', prediction=predicted_class)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
