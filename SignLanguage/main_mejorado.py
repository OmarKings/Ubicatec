# Asistente TEC Optimizado - 3 modos: Manos, Voz, Teclado
# VERSIÓN CORREGIDA - Detección de cámara mejorada + MQTT integrado

import os
import warnings
warnings.filterwarnings("ignore")

import cv2
import mediapipe as mp
import numpy as np
import pickle
import joblib
import time
import queue as qmod
from collections import deque, Counter

import pyttsx3
from googletrans import Translator
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0
import speech_recognition as sr
import unicodedata

# MQTT para comunicación con robot
import paho.mqtt.client as mqtt

# ====
#    UTILIDAD LIMPIAR TEXTO
# ====
def clean_text(s):
    """Quita acentos, tildes y ñ/Ñ"""
    s = s.replace("ñ", "n").replace("Ñ", "N")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


# ====
#    CONFIG DE PANTALLA Y UI
# ====
WIN_W, WIN_H = 1920, 1080
CONSOLE_H = int(WIN_H * 0.30)  
CAM_H = WIN_H - CONSOLE_H    

WHITE = (255, 255, 255)
TEC_BLUE = (0, 51, 160)
TEC_BLUE_LIGHT = (52, 107, 199)

console_lines = deque(maxlen=500)
console_offset = 0
last_console_len = 0


def add_log(src, msg):
    line = f"[{clean_text(src)}] {clean_text(msg)}"
    console_lines.append(line)
    print(line)
    # Añadir línea en blanco para mejor legibilidad
    if src in ["SENAS", "VOZ", "TECLADO", "TRAD", "SISTEMA", "ROBOT"]:
        console_lines.append("")


# ====
#    MQTT CONFIGURACIÓN
# ====
MQTT_SERVER = "958d6cbfe5ce4f0581776ff12f2049b4.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USER = "Omarkings"
MQTT_PASS = "567129Aa"

TOPIC_MANUAL = "omar/manual/accion"
TOPIC_ESTADO = "omar/robot/estado"
TOPIC_GPS = "omar/robot/gps"
TOPIC_NAVIGATION = "omar/robot/navigation"

mqtt_client = None
robot_estado = "Desconocido"
robot_gps = "Sin GPS"
movimiento_habilitado = False  # Control de movimiento por señas


def on_mqtt_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        add_log("MQTT", "Conectado a HiveMQ")
        client.subscribe(TOPIC_ESTADO)
        client.subscribe(TOPIC_GPS)
    else:
        add_log("MQTT", f"Error de conexion: {rc}")


def on_mqtt_message(client, userdata, msg):
    global robot_estado, robot_gps
    try:
        payload = msg.payload.decode('utf-8')
        
        if msg.topic == TOPIC_ESTADO:
            robot_estado = payload.replace("estado: ", "")
            add_log("ROBOT", f"Estado: {robot_estado}")
        
        elif msg.topic == TOPIC_GPS:
            robot_gps = payload
            # No loguear GPS constantemente para no saturar consola
    except Exception as e:
        add_log("MQTT", f"Error procesando mensaje: {e}")


def init_mqtt():
    global mqtt_client
    try:
        mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        mqtt_client.username_pw_set(MQTT_USER, MQTT_PASS)
        mqtt_client.tls_set()
        mqtt_client.on_connect = on_mqtt_connect
        mqtt_client.on_message = on_mqtt_message
        
        mqtt_client.connect(MQTT_SERVER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        add_log("SISTEMA", "MQTT iniciado")
        return True
    except Exception as e:
        add_log("SISTEMA", f"Error MQTT: {e}")
        return False


def send_robot_command(cmd):
    """Envía comando al robot solo si movimiento está habilitado"""
    global movimiento_habilitado
    
    if not movimiento_habilitado:
        add_log("ROBOT", f"Movimiento deshabilitado. Comando '{cmd}' ignorado")
        return
    
    if mqtt_client:
        try:
            mqtt_client.publish(TOPIC_MANUAL, cmd)
            add_log("ROBOT", f"Comando enviado: {cmd}")
        except Exception as e:
            add_log("ROBOT", f"Error enviando comando: {e}")


def send_navigation_command(destino):
    """Envía comando de navegación al robot"""
    if mqtt_client:
        try:
            mqtt_client.publish(TOPIC_NAVIGATION, destino)
            add_log("ROBOT", f"Navegacion: {destino}")
        except Exception as e:
            add_log("ROBOT", f"Error navegacion: {e}")


# ====
#    MODELO DE SENAS
# ====
model = None
classes = None

def load_sign_model():
    global model, classes
    model_file = "model.joblib" if os.path.exists("model.joblib") else "model.p"
    if not os.path.exists(model_file):
        add_log("SISTEMA", "No se encontro modelo de senas.")
        return
    
    try:
        if model_file.endswith(".joblib"):
            data = joblib.load(model_file)
            model = data["model"]
            classes = data.get("classes", model.classes_)
        else:
            with open(model_file, "rb") as f:
                data = pickle.load(f)
            model = data["model"]
            classes = model.classes_

        # Reemplazar alfabeto por uno sin ñ
        letters = [chr(65 + i) for i in range(26)]
        classes = np.array([clean_text(c) for c in letters])

        add_log("SISTEMA", "Modelo de senas cargado.")
    except Exception as e:
        add_log("ERROR", f"Error cargando modelo: {e}")

load_sign_model()


# ====
#    TRADUCCION Y TTS
# ====
translator = Translator()

def decir(texto):
    texto = clean_text(texto)
    if not texto.strip():
        return
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        engine.say(texto)
        engine.runAndWait()
    except:
        pass

def auto_translate(frase):
    frase = clean_text(frase.strip())
    if not frase:
        return
    try:
        lang = detect(frase)
    except:
        add_log("TRAD", "Idioma desconocido")
        return

    if lang == "es":
        add_log("TRAD", "Esp -> no traducido")
        return

    try:
        tr = translator.translate(frase, dest="es")
        tr_text = clean_text(tr.text)
        add_log("TRAD", f"{lang.upper()} -> ES: {tr_text}")
    except Exception as e:
        add_log("TRAD", f"Error: {e}")


# ====
#    PROCESAMIENTO DE COMANDOS DE NAVEGACIÓN
# ====
def procesar_navegacion(frase):
    """Detecta comandos de navegación en la frase"""
    frase_lower = frase.lower()
    
    # Detectar destinos
    if "tim" in frase_lower or "horton" in frase_lower:
        send_navigation_command("NAVEGAR_timhortons")
        add_log("NAVEGACION", "Navegando a Tim Hortons")
        return True
    
    elif "biblioteca" in frase_lower or "biblio" in frase_lower:
        send_navigation_command("NAVEGAR_biblioteca")
        add_log("NAVEGACION", "Navegando a Biblioteca")
        return True
    
    return False


# ====
#    MICROFONO Y WAKEWORD
# ====
voice_events = qmod.Queue()
voice_active = False
current_voice_phrase = ""
current_mode = "manos"  # manos, voz, teclado
wakeword_detected = False  # Nueva bandera


def detect_mic_index():
    mics = sr.Microphone.list_microphone_names()
    for i, n in enumerate(mics):
        if "realtek" in n.lower() or "predet" in n.lower():
            return i
    return 0


MIC_INDEX = detect_mic_index()


def mic_callback(recognizer, audio):
    """Reconoce frases en segundo plano sin bloquear."""
    global current_mode, voice_active, current_voice_phrase, wakeword_detected
    
    try:
        txt = recognizer.recognize_google(audio, language="es-MX").lower()
        txt = clean_text(txt)
        
        # Solo loguear si es wakeword o si ya estamos en modo voz
        if "ayuda" in txt:
            wakeword_detected = True
            current_mode = "voz"
            voice_active = True
            current_voice_phrase = ""
            add_log("VOZ", "Wakeword AYUDA detectada. Modo voz ACTIVO.")
            return
        
        # Si ya estamos en modo voz, acumula texto
        if current_mode == "voz" and voice_active and wakeword_detected:
            add_log("VOZ", f"Escuchado: {txt}")
            voice_events.put(("voice", txt))
            
    except sr.UnknownValueError:
        return  # No se entendió el audio
    except sr.RequestError as e:
        add_log("VOZ", f"Error de servicio: {e}")
        return
    except Exception as e:
        add_log("VOZ", f"Error inesperado: {e}")
        return


def start_mic_background():
    try:
        rec = sr.Recognizer()
        rec.energy_threshold = 400  # Aumentado para evitar activaciones falsas
        rec.dynamic_energy_threshold = True
        rec.pause_threshold = 1.0  # Más pausa para capturar "ayuda" completo
        
        mic = sr.Microphone(device_index=MIC_INDEX)
        add_log("VOZ", "Ajustando microfono...")
        
        with mic as source:
            rec.adjust_for_ambient_noise(source, duration=2)
        
        add_log("VOZ", "Microfono listo. Di 'ayuda' para activar.")
        rec.listen_in_background(mic, mic_callback, phrase_time_limit=5)
        
    except Exception as e:
        add_log("VOZ", f"Error iniciando microfono: {e}")


# ====
#    ESTADO DE SENAS
# ====
current_sign_word = ""
last_letter = "?"
last_conf = 0.0
last_valid = 0
pred_buffer = deque(maxlen=10)
NO_DET_TIMEOUT = 2.0
LETTER_HOLD = 1.5
last_det = 0

USE_LEFT_HAND_ONLY = True
ENABLE_FLIP = True
MIRROR_LANDMARKS = True

# Estado de teclado
current_text_keyboard = ""


# ============================================
#    CONFIG DE CÁMARA – FORZAR CAM 1 (TU CAMARA)
# ============================================

FORCED_CAM_INDEX = 1   # <-- ESTA ES TU CÁMARA QUE SÍ FUNCIONA


def find_working_camera():
    """Abre exclusivamente la cámara 1 usando CAP_DSHOW."""
    add_log("SISTEMA", f"Intentando abrir cámara fija: {FORCED_CAM_INDEX}")

    # Intentar abrir usando DSHOW (Windows)
    cap = cv2.VideoCapture(FORCED_CAM_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        add_log("ERROR", f"No se pudo abrir la camara {FORCED_CAM_INDEX}")
        return None, -1

    # Dar tiempo para que la cámara active el sensor
    add_log("SISTEMA", "Inicializando cámara...")
    ok = False
    for _ in range(15):
        ret, frame = cap.read()
        if ret and frame is not None:
            ok = True
            break
        time.sleep(0.05)

    if not ok:
        add_log("ERROR", f"La camara {FORCED_CAM_INDEX} abrió pero no entrega frames")
        cap.release()
        return None, -1

    add_log("SISTEMA", f"Camara {FORCED_CAM_INDEX} funcionando correctamente")
    return cap, FORCED_CAM_INDEX

# ====
#    UI (sin PIL)
# ====
def put_text(img, text, x, y, scale=0.8, color=(255,255,255), thickness=2):
    text = clean_text(text)
    cv2.putText(img, text, (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_console(img):
    global console_offset, last_console_len

    cv2.rectangle(img, (0, CAM_H), (WIN_W, WIN_H), (0, 0, 0), -1)

    lines = list(console_lines)
    n = len(lines)

    line_height = 30
    padding_top = 25
    padding_left = 40
    visible_lines = (CONSOLE_H - padding_top - 10) // line_height

    if console_offset == 0 and n != last_console_len:
        last_console_len = n

    max_offset = max(0, n - visible_lines)
    console_offset = min(console_offset, max_offset)

    start = max(0, n - visible_lines - console_offset)
    end = start + visible_lines

    y = CAM_H + padding_top

    for line in lines[start:end]:
        put_text(img, line, padding_left, y, scale=0.65, color=(200,200,200), thickness=1)
        y += line_height

    return img


def place_cam(frame, canvas):
    h, w, _ = frame.shape
    scale = min(WIN_W / w, CAM_H / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh))

    xo = (WIN_W - nw) // 2
    yo = (CAM_H - nh) // 2

    canvas[yo:yo + nh, xo:xo + nw] = resized
    return canvas, xo


# ====
#    LOOP PRINCIPAL
# ====
def main():
    global current_sign_word, last_letter, last_conf, last_valid
    global last_det, voice_active, current_voice_phrase, console_offset
    global current_mode, current_text_keyboard, movimiento_habilitado
    global wakeword_detected

    # Iniciar MQTT
    init_mqtt()
    
    # Iniciar micrófono
    start_mic_background()

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mpDraw = mp.solutions.drawing_utils

    # Buscar cámara funcional
    cap, cam_idx = find_working_camera()
    if cap is None:
        add_log("ERROR", "No se puede continuar sin camara")
        return
    
    # Configurar cámara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    add_log("SISTEMA", "Asistente TEC optimizado iniciado.")
    add_log("SISTEMA", "Modo inicial: MANOS")
    add_log("SISTEMA", "Movimiento del robot: DESHABILITADO")

    frame_count = 0
    last_frame_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                add_log("ERROR", "No se pudo leer frame de la camara")
                time.sleep(0.1)
                continue

            frame_count += 1

            if ENABLE_FLIP:
                frame = cv2.flip(frame, 1)

            status = "Sin mano"

            # ====
            #    PROCESADO DE SENAS (solo en modo manos)
            # ====
            if model is not None and current_mode == "manos":
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    detection_time = time.time()
                    if detection_time - last_det > NO_DET_TIMEOUT:
                        pred_buffer.clear()

                    for handLms, handedness in zip(results.multi_hand_landmarks,
                                                    results.multi_handedness):

                        label = handedness.classification[0].label
                        if ENABLE_FLIP:
                            label = "Right" if label == "Left" else "Left"

                        if USE_LEFT_HAND_ONLY and label != "Left":
                            continue

                        lm = np.array([[lm.x, lm.y] for lm in handLms.landmark])

                        if MIRROR_LANDMARKS and label == "Left":
                            lm[:,0] = 1 - lm[:,0]

                        lm -= lm[0]
                        maxv = np.max(np.abs(lm))
                        if maxv != 0:
                            lm /= maxv

                        lm = lm.flatten()
                        if lm.shape != (42,):
                            continue

                        pred = model.predict([lm])[0]
                        conf = float(np.max(model.predict_proba([lm])[0]) * 100)
                        pred_buffer.append(pred)
                        last_det = detection_time

                        min_frames = 7

                        if len(pred_buffer) >= min_frames:
                            most = Counter(pred_buffer).most_common(1)
                            if most[0][1] >= 7 and conf > 60:
                                last_letter = clean_text(most[0][0])
                                last_conf = conf
                                last_valid = time.time()
                                status = f"Senas: {last_letter} ({int(conf)}%)"

                                if not current_sign_word or current_sign_word[-1] != last_letter:
                                    current_sign_word += last_letter
                                    add_log("SENAS", f"Parcial: {current_sign_word}")
                        else:
                            status = "Procesando gesto..."

                        mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

                else:
                    if time.time() - last_valid > LETTER_HOLD:
                        last_letter = "?"


            # ====
            #    PROCESADO DE VOZ
            # ====
            while not voice_events.empty():
                t, text = voice_events.get()
                text = clean_text(text)

                if current_mode == "voz" and voice_active:
                    current_voice_phrase += " " + text
                    add_log("VOZ", f"Parcial: {current_voice_phrase}")


            # ====
            #    DIBUJAR UI
            # ====
            canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
            canvas, xo = place_cam(frame, canvas)

            # Indicador de modo con color
            mode_color = (100, 255, 100) if current_mode == "manos" else (255, 200, 0) if current_mode == "voz" else (100, 200, 255)
            
            # Estado del robot
            mov_color = (0, 255, 0) if movimiento_habilitado else (0, 0, 255)
            
            put_text(canvas, status, 40, 60, scale=1.0, color=TEC_BLUE_LIGHT, thickness=2)
            put_text(canvas, f"Modo: {current_mode.upper()}", 40, 100, scale=0.9, color=mode_color, thickness=2)
            put_text(canvas, f"Movimiento: {'HABILITADO' if movimiento_habilitado else 'DESHABILITADO'}", 
                     40, 140, scale=0.8, color=mov_color, thickness=2)
            put_text(canvas, f"Robot: {robot_estado}", 40, 180, scale=0.7, color=(200,200,200))
            put_text(canvas, f"Senas: {current_sign_word}", 40, 220, scale=0.8)
            put_text(canvas, f"Voz: {current_voice_phrase}", 40, 260, scale=0.8)
            put_text(canvas, f"Teclado: {current_text_keyboard}", 40, 300, scale=0.8)

            put_text(canvas, "M=Mano | 'ayuda'/V=Voz | T=Teclado | Enter=Confirmar | Q/Esc=Salir",
                     40, 340, scale=0.6, color=(180,180,180), thickness=1)

            canvas = draw_console(canvas)

            cv2.imshow("Asistente TEC Optimizado", canvas)
            key = cv2.waitKey(1) & 0xFFFF

            # Salir
            if key in (ord("q"), 27):
                break

            # Cambiar de modo con teclas
            if key in (ord("m"), ord("M")):
                current_mode = "manos"
                add_log("SISTEMA", "Modo cambiado a MANOS")
            elif key in (ord("v"), ord("V")):
                current_mode = "voz"
                voice_active = True
                wakeword_detected = True
                current_voice_phrase = ""
                add_log("SISTEMA", "Modo cambiado a VOZ (forzado por teclado)")
            elif key in (ord("t"), ord("T")):
                current_mode = "teclado"
                current_text_keyboard = ""
                add_log("SISTEMA", "Modo cambiado a TECLADO (escribe y Enter)")

            # Captura de texto en modo teclado
            if current_mode == "teclado":
                if 32 <= key <= 126:
                    current_text_keyboard += chr(key)
                elif key == 8:  # Backspace
                    current_text_keyboard = current_text_keyboard[:-1]

            # scroll console ↑ ↓
            if key == 2490368:
                console_offset += 1
            elif key == 2621440:
                console_offset = max(0, console_offset - 1)

            # ENTER = confirmar según el modo actual
            if key in (13, 10):
                if current_mode == "manos" and current_sign_word.strip():
                    palabra = current_sign_word.strip().upper()
                    add_log("SENAS", f"Confirmada: {palabra}")
                    
                    # Detectar comandos especiales
                    if palabra == "MOVER":
                        movimiento_habilitado = True
                        add_log("ROBOT", "Movimiento HABILITADO")
                    elif palabra == "PARAR":
                        movimiento_habilitado = False
                        send_robot_command("QUIETO")
                        add_log("ROBOT", "Movimiento DESHABILITADO")
                    else:
                        # Intentar navegación
                        if not procesar_navegacion(palabra):
                            auto_translate(palabra)
                    
                    current_sign_word = ""

                elif current_mode == "voz" and current_voice_phrase.strip():
                    frase = current_voice_phrase.strip()
                    add_log("VOZ", f"Confirmada: {frase}")
                    
                    # Intentar navegación primero
                    if not procesar_navegacion(frase):
                        auto_translate(frase)
                    
                    current_voice_phrase = ""
                    voice_active = False
                    wakeword_detected = False
                    current_mode = "manos"
                    add_log("SISTEMA", "Se regresa a modo MANOS")

                elif current_mode == "teclado" and current_text_keyboard.strip():
                    texto = current_text_keyboard.strip()
                    add_log("TECLADO", f"Confirmada: {texto}")
                    
                    # Intentar navegación primero
                    if not procesar_navegacion(texto):
                        auto_translate(texto)
                    
                    current_text_keyboard = ""

    except KeyboardInterrupt:
        add_log("SISTEMA", "Interrupcion detectada (Ctrl+C)")
    except Exception as e:
        add_log("ERROR", f"Error inesperado: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Limpieza de recursos
        add_log("SISTEMA", "Cerrando sistema...")
        
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
            add_log("SISTEMA", "MQTT desconectado")
        
        cap.release()
        cv2.destroyAllWindows()
        add_log("SISTEMA", "Sistema cerrado correctamente")


if __name__ == "__main__":
    main()