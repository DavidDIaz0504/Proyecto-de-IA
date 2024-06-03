import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Cargar el modelo
model = tf.keras.models.load_model('Model.h5')

# Función para predecir la clase de una imagen
def predict(image):
    img_array = np.array(image.resize((224, 224))) / 255.0  # Ajustar el tamaño de la imagen según sea necesario
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    confidence = np.max(predictions)
    return class_idx, confidence

st.title('Product Scanner')

# Inicializar la cámara
camera = cv2.VideoCapture(0)

if st.button('Capture'):
    # Capturar un fotograma de la cámara
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Mostrar el fotograma en la interfaz de Streamlit
    st.image(frame, channels="RGB", use_column_width=True)
    
    # Convertir la imagen capturada en un objeto de imagen de PIL
    captured_image = Image.fromarray(frame)
    
    # Predecir la clase de la imagen capturada
    class_idx, confidence = predict(captured_image)
    
    # Mostrar la clasificación y la confianza
    st.write(f'Class: {class_idx}, Confidence: {confidence:.2f}')

# Liberar la cámara
camera.release()
cv2.destroyAllWindows()
