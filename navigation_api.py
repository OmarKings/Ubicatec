from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import paho.mqtt.client as mqtt
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'tu_clave_secreta_aqui_12345'
socketio = SocketIO(app, cors_allowed_origins="*")

# ======================================================
# CONFIGURACIÓN MQTT
# ======================================================
MQTT_BROKER = "958d6cbfe5ce4f0581776ff12f2049b4.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USER = "Omarkings"
MQTT_PASS = "567129Aa"

# Topics
TOPIC_GPS_ROBOT = "omar/robot/gps"
TOPIC_ESTADO_ROBOT = "omar/robot/estado"
TOPIC_GEOCERCA_ROBOT = "omar/robot/geocerca"
TOPIC_MANUAL = "omar/manual/accion"
TOPIC_NAVIGATION = "omar/robot/navigation"
TOPIC_ALERTA = "kinect/alerta_led"

# Topics de fallback PC
TOPIC_FALLBACK_PC = "omar/pc/fallback"
TOPIC_GEOCERCA_PC = "omar/pc/geocerca"
TOPIC_GPS_PC = "omar/pc/gps"

# ======================================================
# GEOCERCA POLIGONAL ITESMQ (misma que en ESP32)
# ======================================================
GEO_LAT = [
    20.613123, 20.613356, 20.613529, 20.613851, 20.613936,
    20.614039, 20.614099, 20.614039, 20.613944, 20.613880,
    20.613273, 20.613145, 20.613189, 20.613327, 20.613292
]

GEO_LON = [
    -100.403724, -100.403872, -100.403560, -100.403539, -100.403598,
    -100.403571, -100.403405, -100.403379, -100.403297, -100.403461,
    -100.403356, -100.403289, -100.403187, -100.403243, -100.403336
]

def punto_en_poligono(lat, lon):
    """Ray casting algorithm para verificar si un punto está dentro del polígono"""
    dentro = False
    n = len(GEO_LAT)
    j = n - 1
    
    for i in range(n):
        if ((GEO_LAT[i] > lat) != (GEO_LAT[j] > lat)) and \
           (lon < (GEO_LON[j] - GEO_LON[i]) * (lat - GEO_LAT[i]) / (GEO_LAT[j] - GEO_LAT[i]) + GEO_LON[i]):
            dentro = not dentro
        j = i
    
    return dentro

# ======================================================
# VARIABLES GLOBALES
# ======================================================
robot_data = {
    "gps": None,
    "estado": "desconocido",
    "geocerca": "desconocido",
    "last_gps_time": 0
}

pc_data = {
    "gps": None,
    "geocerca": "desconocido",
    "fallback_activo": False
}

TIMEOUT_GPS_ROBOT = 10  # segundos sin GPS del robot para activar fallback

# ======================================================
# CLIENTE MQTT
# ======================================================
mqtt_client = mqtt.Client()
mqtt_client.username_pw_set(MQTT_USER, MQTT_PASS)
mqtt_client.tls_set()

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✓ Conectado a MQTT broker")
        client.subscribe([
            (TOPIC_GPS_ROBOT, 0),
            (TOPIC_ESTADO_ROBOT, 0),
            (TOPIC_GEOCERCA_ROBOT, 0),
            (TOPIC_GPS_PC, 0),
            (TOPIC_FALLBACK_PC, 0),
            (TOPIC_GEOCERCA_PC, 0)
        ])
    else:
        print(f"✗ Error de conexión MQTT: {rc}")

def on_message(client, userdata, msg):
    global robot_data, pc_data
    
    topic = msg.topic
    payload = msg.payload.decode('utf-8')
    
    if topic == TOPIC_GPS_ROBOT:
        robot_data["gps"] = payload
        robot_data["last_gps_time"] = time.time()
        print(f"GPS Robot recibido: {payload}")
        
        # Parsear coordenadas
        try:
            coords = payload.split("|")[0]  # "lat,lon|fecha hora"
            lat, lon = map(float, coords.split(","))
            
            # Emitir a frontend
            socketio.emit('robot_gps', {
                'lat': lat,
                'lon': lon,
                'raw': payload
            })
        except Exception as e:
            print(f"Error parseando GPS: {e}")
    
    elif topic == TOPIC_ESTADO_ROBOT:
        robot_data["estado"] = payload
        print(f"Estado Robot: {payload}")
        socketio.emit('robot_estado', {'estado': payload})
    
    elif topic == TOPIC_GEOCERCA_ROBOT:
        robot_data["geocerca"] = payload
        print(f"Geocerca Robot: {payload}")
        socketio.emit('robot_geocerca', {'geocerca': payload})
    
    elif topic == TOPIC_GPS_PC:
        pc_data["gps"] = payload
        print(f"GPS PC recibido: {payload}")
    
    elif topic == TOPIC_FALLBACK_PC:
        pc_data["fallback_activo"] = (payload == "ON")
        print(f"Fallback PC: {payload}")
        socketio.emit('fallback_status', {'activo': pc_data["fallback_activo"]})
    
    elif topic == TOPIC_GEOCERCA_PC:
        pc_data["geocerca"] = payload
        print(f"Geocerca PC: {payload}")
        socketio.emit('pc_geocerca_status', {'geocerca': payload})

mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

# ======================================================
# HILO DE MONITOREO DE FALLBACK (OPCIONAL - REDUNDANCIA)
# ======================================================
def monitorear_fallback():
    """
    Hilo que verifica si el robot dejó de enviar GPS y activa fallback de PC
    NOTA: El frontend (index.html) ya hace esto, pero lo dejamos por redundancia
    """
    global robot_data, pc_data
    
    while True:
        time.sleep(3)  # Revisar cada 3 segundos
        
        tiempo_sin_gps = time.time() - robot_data["last_gps_time"]
        
        # Activar fallback si no hay GPS del robot
        if tiempo_sin_gps > TIMEOUT_GPS_ROBOT and not pc_data["fallback_activo"]:
            print("⚠️  SERVIDOR: Activando fallback de PC (sin GPS del robot)")
            pc_data["fallback_activo"] = True
            mqtt_client.publish(TOPIC_FALLBACK_PC, "ON")
            socketio.emit('fallback_status', {'activo': True})
            
        # Desactivar fallback si vuelve el GPS del robot
        elif tiempo_sin_gps <= TIMEOUT_GPS_ROBOT and pc_data["fallback_activo"]:
            print("✓ SERVIDOR: Desactivando fallback de PC (robot volvió)")
            pc_data["fallback_activo"] = False
            mqtt_client.publish(TOPIC_FALLBACK_PC, "OFF")
            socketio.emit('fallback_status', {'activo': False})

# ======================================================
# RUTAS FLASK
# ======================================================
@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/api/robot/status')
def robot_status():
    """Devuelve el estado actual del robot"""
    return jsonify(robot_data)

@app.route('/api/pc/status')
def pc_status():
    """Devuelve el estado actual de la PC (fallback)"""
    return jsonify(pc_data)

@app.route('/api/comando', methods=['POST'])
def enviar_comando():
    """Envía comando manual al robot por MQTT"""
    data = request.json
    comando = data.get('comando', '').upper()
    
    comandos_validos = ['ADELANTE', 'ATRAS', 'IZQUIERDA', 'DERECHA', 'QUIETO', 'ALERTA', 'AVANZAR']
    
    if comando in comandos_validos:
        mqtt_client.publish(TOPIC_MANUAL, comando)
        return jsonify({'success': True, 'comando': comando})
    else:
        return jsonify({'success': False, 'error': 'Comando inválido'}), 400

# ======================================================
# SOCKETIO EVENTS
# ======================================================
@socketio.on('connect')
def handle_connect():
    print('Cliente web conectado')
    emit('connection_response', {'data': 'Conectado al servidor'})

@socketio.on('pc_gps_update')
def handle_pc_gps(data):
    """Recibe la ubicación GPS de la PC desde el navegador"""
    global pc_data
    
    lat = data.get('lat')
    lon = data.get('lon')
    
    if lat is None or lon is None:
        return
    
    pc_data["gps"] = f"{lat},{lon}"
    
    # Evaluar geocerca
    dentro = punto_en_poligono(lat, lon)
    estado_geo = "DENTRO" if dentro else "FUERA"
    pc_data["geocerca"] = estado_geo
    
    print(f"GPS PC: ({lat}, {lon}) -> Geocerca: {estado_geo}")
    
    # Publicar por MQTT
    mqtt_client.publish(TOPIC_GPS_PC, f"{lat},{lon}")
    mqtt_client.publish(TOPIC_GEOCERCA_PC, estado_geo)
    
    # Emitir a todos los clientes
    emit('pc_geocerca', {
        'lat': lat,
        'lon': lon,
        'geocerca': estado_geo,
        'fallback_activo': pc_data["fallback_activo"]
    }, broadcast=True)

# ======================================================
# INICIALIZACIÓN
# ======================================================
def iniciar_mqtt():
    """Conecta al broker MQTT en un hilo separado"""
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        print("✓ Cliente MQTT iniciado")
    except Exception as e:
        print(f"✗ Error al conectar MQTT: {e}")

if __name__ == '__main__':
    print("=" * 60)
    print("🤖 SISTEMA ROBOT ITESMQ CON FALLBACK PC")
    print("=" * 60)
    
    # Iniciar MQTT
    iniciar_mqtt()
    
    # Iniciar hilo de monitoreo de fallback (opcional, redundancia)
    hilo_fallback = threading.Thread(target=monitorear_fallback, daemon=True)
    hilo_fallback.start()
    
    # Iniciar servidor Flask
    print("🚀 Servidor iniciado en http://localhost:5000")
    print("=" * 60)
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)