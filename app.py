from flask import Flask, render_template, send_from_directory
from flask_mqtt import Mqtt
from flask_socketio import SocketIO, emit
import tensorflow as tf
import numpy as np
import json
from datetime import datetime
import os

# Configurar Flask, MQTT y SocketIO
app = Flask(__name__, static_folder='public', static_url_path='/public')
app.config['MQTT_BROKER_URL'] = 'a632c67843b6481cb94e93b8053f29dd.s1.eu.hivemq.cloud'
app.config['MQTT_BROKER_PORT'] = 8883
app.config['MQTT_USERNAME'] = 'miye'
app.config['MQTT_PASSWORD'] = 'Miye1234'
app.config['MQTT_KEEPALIVE'] = 60
app.config['MQTT_TLS_ENABLED'] = True
mqtt = Mqtt(app)
socketio = SocketIO(app)

# Variables globales
gate_status = 'CERRADA'
MQTT_TOPIC = "sensores/temperatura_humedad"

# Funciones de normalización y escalado
def scale_and_normalize(value, scale, min_val, max_val):
    scaled = value * scale
    return (scaled - min_val) / (max_val - min_val)

# Datos de entrenamiento y preparación del modelo
training_data = [
    # Temperaturas <=24°C (Salida: 0)

    {'input': [scale_and_normalize(20.00, 0.7, 0, 40), scale_and_normalize(30.00, 0.3, 0, 100)], 'output': [0]},
    {'input': [scale_and_normalize(18.00, 0.7, 0, 40), scale_and_normalize(35.00, 0.3, 0, 100)], 'output': [0]},
    {'input': [scale_and_normalize(25.00, 0.7, 0, 40), scale_and_normalize(95.00, 0.3, 0, 100)], 'output': [1]},
    {'input': [scale_and_normalize(28.00, 0.7, 0, 40), scale_and_normalize(100.00, 0.3, 0, 100)], 'output': [1]},
]

# Preparar datos para TensorFlow
xs = np.array([d['input'] for d in training_data], dtype=np.float32)
ys = np.array([d['output'] for d in training_data], dtype=np.float32)

# Crear el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(2,), activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(xs, ys, epochs=200, verbose=1, validation_split=0.2, shuffle=True)

# Rutas de la aplicación
@app.route('/')
def index():
    return send_from_directory('public', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('public', path)

# Manejadores de MQTT
@mqtt.on_connect()
def handle_connect(client, userdata, flags, rc):
    mqtt.subscribe(MQTT_TOPIC)

@mqtt.on_message()
def handle_mqtt_message(client, userdata, message):
    global gate_status
    payload = json.loads(message.payload.decode())
    temperatura = float(payload['temperatura'])
    humedad = float(payload['humedad'])

    # Normalizar y escalar los datos
    temperatura_norm = scale_and_normalize(temperatura, 0.7, 0, 21.7)
    humedad_norm = scale_and_normalize(humedad, 0.3, 0, 30)

    # Obtener respuesta del modelo
    input_tensor = np.array([[temperatura_norm, humedad_norm]], dtype=np.float32)
    output_tensor = model.predict(input_tensor)
    respuesta = int(round(output_tensor[0][0]))

    # Actualizar estado de la compuerta basado en temperatura
    if temperatura > 25:
        gate_status = 'ABIERTA'
    else:
        gate_status = 'CERRADA'

    # Emitir datos a los clientes conectados
    socketio.emit('datos', {
        'temperatura': temperatura,
        'humedad': humedad,
        'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'respuesta': respuesta,
        'gateStatus': gate_status
    })

# Manejadores de Socket.IO
@socketio.on('connect')
def handle_socket_connect():
    emit('gateStatus', gate_status)

# Iniciar el servidor
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=3000, debug=True)
