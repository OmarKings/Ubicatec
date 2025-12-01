#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de GPS Fallback para Robot Autónomo
Proporciona coordenadas GPS desde la computadora cuando el ESP32 no tiene señal
Versión compatible con Windows (sin emojis)
"""

import paho.mqtt.client as mqtt
import json
import time
import sys
import io

# Configurar salida para Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ====================================
# CONFIGURACIÓN MQTT
# ====================================
MQTT_SERVER = "958d6cbfe5ce4f0581776ff12f2049b4.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USER = "Omarkings"
MQTT_PASS = "567129Aa"

TOPIC_GPS_FALLBACK = "omar/gps/fallback"
TOPIC_GPS_ROBOT = "omar/robot/gps"
TOPIC_ESTADO = "omar/robot/estado"

# ====================================
# VARIABLES GLOBALES
# ====================================
gps_robot_valido = False
ultima_actualizacion_robot = 0
TIMEOUT_GPS_ROBOT = 15  # segundos

# Coordenadas de fallback (puedes obtenerlas de Google Maps o tu teléfono)
# Por defecto, centro del ITESMQ
fallback_lat = 20.613500
fallback_lon = -100.403000

# ====================================
# FUNCIONES MQTT
# ====================================
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("[OK] Conectado a MQTT")
        client.subscribe(TOPIC_GPS_ROBOT)
        client.subscribe(TOPIC_ESTADO)
        print(f"[INFO] Escuchando GPS del robot en: {TOPIC_GPS_ROBOT}")
    else:
        print(f"[ERROR] Error de conexion: {rc}")

def on_message(client, userdata, msg):
    global gps_robot_valido, ultima_actualizacion_robot
    
    try:
        if msg.topic == TOPIC_GPS_ROBOT:
            data = json.loads(msg.payload.decode('utf-8'))
            
            if data.get('valido', False):
                gps_robot_valido = True
                ultima_actualizacion_robot = time.time()
                print(f"[OK] GPS del robot valido: {data['lat']:.6f}, {data['lon']:.6f}")
            else:
                gps_robot_valido = False
                print("[WARN] GPS del robot no valido")
        
        elif msg.topic == TOPIC_ESTADO:
            estado = msg.payload.decode('utf-8')
            if "GPS no disponible" in estado or "Esperando GPS" in estado:
                print("[WARN] Robot solicita GPS fallback")
    
    except Exception as e:
        print(f"[ERROR] Error procesando mensaje: {e}")

def publicar_gps_fallback(client, lat, lon):
    """Publica coordenadas GPS de fallback"""
    data = {
        "lat": lat,
        "lon": lon,
        "fallback": True
    }
    
    mensaje = json.dumps(data)
    client.publish(TOPIC_GPS_FALLBACK, mensaje)
    print(f"[GPS] Fallback enviado: {lat:.6f}, {lon:.6f}")

# ====================================
# OBTENER GPS DE LA COMPUTADORA
# ====================================
def obtener_gps_computadora():
    """
    Intenta obtener GPS de la computadora.
    Métodos disponibles:
    1. Geolocalización por IP (menos preciso)
    2. GPS USB conectado a la computadora
    3. Coordenadas manuales (fallback)
    """
    
    # MÉTODO 1: Usar geolocalización por IP (requiere geocoder)
    try:
        import geocoder
        g = geocoder.ip('me')
        if g.ok:
            print(f"[GPS] Obtenido por IP: {g.latlng}")
            return g.latlng[0], g.latlng[1]
    except ImportError:
        pass  # Módulo no instalado
    except Exception as e:
        print(f"[WARN] Error obteniendo GPS por IP: {e}")
    
    # MÉTODO 2: Usar GPS USB (requiere gpsd)
    try:
        from gps import gps, WATCH_ENABLE
        session = gps(mode=WATCH_ENABLE)
        report = session.next()
        if report['class'] == 'TPV':
            lat = getattr(report, 'lat', 0.0)
            lon = getattr(report, 'lon', 0.0)
            if lat != 0.0 and lon != 0.0:
                print(f"[GPS] Obtenido de USB: {lat}, {lon}")
                return lat, lon
    except ImportError:
        pass  # Módulo no instalado
    except Exception as e:
        print(f"[WARN] Error obteniendo GPS USB: {e}")
    
    # MÉTODO 3: Coordenadas manuales (fallback final)
    print(f"[GPS] Usando coordenadas predeterminadas: {fallback_lat}, {fallback_lon}")
    return fallback_lat, fallback_lon

# ====================================
# FUNCIÓN PRINCIPAL
# ====================================
def main():
    global gps_robot_valido, ultima_actualizacion_robot
    
    print("=" * 70)
    print("SISTEMA DE GPS FALLBACK PARA ROBOT LOOK-Y")
    print("=" * 70)
    print()
    print("Este script proporciona coordenadas GPS al robot cuando")
    print("el GPS del ESP32 no tiene senal.")
    print()
    print("Metodos de GPS disponibles:")
    print("  1. Geolocalizacion por IP (requiere: pip install geocoder)")
    print("  2. GPS USB conectado (requiere: pip install gps)")
    print("  3. Coordenadas predeterminadas")
    print()
    print("=" * 70)
    print()
    
    # Configurar cliente MQTT
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.username_pw_set(MQTT_USER, MQTT_PASS)
    client.tls_set()
    client.on_connect = on_connect
    client.on_message = on_message
    
    # Conectar
    try:
        print("[INFO] Conectando a MQTT...")
        client.connect(MQTT_SERVER, MQTT_PORT, 60)
        client.loop_start()
    except Exception as e:
        print(f"[ERROR] Error conectando a MQTT: {e}")
        sys.exit(1)
    
    print()
    print("[OK] Sistema iniciado")
    print("[INFO] Esperando estado del GPS del robot...")
    print()
    print("Presiona Ctrl+C para salir")
    print()
    
    try:
        while True:
            # Verificar si el GPS del robot está inactivo
            tiempo_sin_gps = time.time() - ultima_actualizacion_robot
            
            if not gps_robot_valido or tiempo_sin_gps > TIMEOUT_GPS_ROBOT:
                if tiempo_sin_gps > TIMEOUT_GPS_ROBOT and ultima_actualizacion_robot > 0:
                    print(f"[WARN] Sin GPS del robot por {int(tiempo_sin_gps)}s")
                
                # Obtener GPS de la computadora
                lat, lon = obtener_gps_computadora()
                
                # Publicar GPS fallback
                publicar_gps_fallback(client, lat, lon)
                
                # Esperar 5 segundos antes de enviar de nuevo
                time.sleep(5)
            else:
                # GPS del robot está funcionando, no hacer nada
                time.sleep(2)
    
    except KeyboardInterrupt:
        print()
        print("[INFO] Deteniendo sistema...")
        client.loop_stop()
        client.disconnect()
        print("[OK] Sistema cerrado correctamente")

# ====================================
# CONFIGURACIÓN MANUAL DE COORDENADAS
# ====================================
def configurar_coordenadas_manual():
    """Permite configurar coordenadas manualmente"""
    global fallback_lat, fallback_lon
    
    print()
    print("=" * 70)
    print("CONFIGURACION MANUAL DE COORDENADAS")
    print("=" * 70)
    print()
    print("Puedes obtener tus coordenadas de:")
    print("  - Google Maps (clic derecho > coordenadas)")
    print("  - Tu telefono (app de GPS)")
    print("  - https://www.latlong.net/")
    print()
    
    try:
        lat = float(input("Ingresa latitud (ej: 20.613500): "))
        lon = float(input("Ingresa longitud (ej: -100.403000): "))
        
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            fallback_lat = lat
            fallback_lon = lon
            print()
            print(f"[OK] Coordenadas configuradas: {lat}, {lon}")
            print()
        else:
            print("[ERROR] Coordenadas invalidas")
            return False
    except ValueError:
        print("[ERROR] Formato invalido")
        return False
    
    return True

# ====================================
# MENÚ PRINCIPAL
# ====================================
if __name__ == "__main__":
    print()
    print("=" * 70)
    print("SISTEMA DE GPS FALLBACK PARA ROBOT AUTONOMO")
    print("=" * 70)
    print()
    print("Opciones:")
    print("  1. Iniciar con coordenadas predeterminadas")
    print("  2. Configurar coordenadas manualmente")
    print("  3. Salir")
    print()
    
    opcion = input("Selecciona una opcion (1-3): ").strip()
    
    if opcion == "1":
        main()
    elif opcion == "2":
        if configurar_coordenadas_manual():
            main()
    elif opcion == "3":
        print("Hasta luego")
        sys.exit(0)
    else:
        print("[ERROR] Opcion invalida")
        sys.exit(1)