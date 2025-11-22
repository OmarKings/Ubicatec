# Asistente TEC Optimizado - 3 modos: Manos, Voz, Teclado

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

# =========================================================
#                  UTILIDAD LIMPIAR TEXTO
# =========================================================
def clean_text(s):
    """Quita acentos, tildes y ñ/Ñ"""
    s = s.replace("ñ", "n").replace("Ñ", "N")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


# =========================================================
#                 CONFIG DE PANTALLA Y UI
# =========================================================
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
    if src in ["SENAS", "VOZ", "TECLADO", "TRAD", "SISTEMA"]:
        console_lines.append("")


# =========================================================
#                 MODELO DE SENAS
# =========================================================
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


# =========================================================
#                TRADUCCION Y TTS
# =========================================================
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


# =========================================================
#                MICROFONO Y WAKEWORD
# =========================================================
voice_events = qmod.Queue()
voice_active = False
current_voice_phrase = ""
current_mode = "manos"  # manos, voz, teclado


def detect_mic_index():
    mics = sr.Microphone.list_microphone_names()
    for i, n in enumerate(mics):
        if "realtek" in n.lower() or "predet" in n.lower():
            return i
    return 0


MIC_INDEX = detect_mic_index()


def mic_callback(recognizer, audio):
    """Reconoce frases en segundo plano sin bloquear."""
    global current_mode, voice_active, current_voice_phrase
    
    try:
        txt = recognizer.recognize_google(audio, language="es-MX").lower()
        txt = clean_text(txt)
        add_log("VOZ", f"Escuchado: {txt}")
    except sr.UnknownValueError:
        return  # No se entendió el audio
    except sr.RequestError as e:
        add_log("VOZ", f"Error de servicio (reconocedor Google): {e}")
        return
    except Exception as e:
        add_log("VOZ", f"Error inesperado en mic_callback: {e}")
        return

    # Detectar wakeword "ayuda"
    if "ayuda" in txt:
        current_mode = "voz"
        voice_active = True
        current_voice_phrase = ""
        add_log("VOZ", "Wakeword AYUDA detectada. Modo voz ACTIVO.")
        return

    # Si ya estamos en modo voz, acumula texto
    if current_mode == "voz" and voice_active:
        voice_events.put(("voice", txt))


def start_mic_background():
    try:
        rec = sr.Recognizer()
        rec.energy_threshold = 300  # Sensibilidad del micrófono
        rec.dynamic_energy_threshold = True
        rec.pause_threshold = 0.8  # Pausa entre palabras
        
        mic = sr.Microphone(device_index=MIC_INDEX)
        add_log("VOZ", "Ajustando microfono...")
        
        with mic as source:
            rec.adjust_for_ambient_noise(source, duration=1.5)
        
        add_log("VOZ", "Microfono listo. Di 'ayuda' para activar.")
        rec.listen_in_background(mic, mic_callback, phrase_time_limit=5)
        
    except Exception as e:
        add_log("VOZ", f"Error iniciando microfono: {e}")


# =========================================================
#                 ESTADO DE SENAS
# =========================================================
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


# =========================================================
#                    UI (sin PIL)
# =========================================================
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

    line_height = 30  # Más espacio entre líneas
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


# =========================================================
#                        LOOP PRINCIPAL
# =========================================================
def main():
    global current_sign_word, last_letter, last_conf, last_valid
    global last_det, voice_active, current_voice_phrase, console_offset
    global current_mode, current_text_keyboard

    start_mic_background()

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mpDraw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

    add_log("SISTEMA", "Asistente TEC optimizado iniciado.")
    add_log("SISTEMA", "Modo inicial: MANOS")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if ENABLE_FLIP:
            frame = cv2.flip(frame, 1)

        status = "Sin mano"

        # ===========================
        #      PROCESADO DE SENAS (solo en modo manos)
        # ===========================
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


        # ===========================
        #      PROCESADO DE VOZ
        # ===========================
        while not voice_events.empty():
            t, text = voice_events.get()
            text = clean_text(text)

            if current_mode == "voz" and voice_active:
                current_voice_phrase += " " + text
                add_log("VOZ", f"Parcial: {current_voice_phrase}")


        # ===========================
        #         DIBUJAR UI
        # ===========================
        canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
        canvas, xo = place_cam(frame, canvas)

        # Indicador de modo con color
        mode_color = (100, 255, 100) if current_mode == "manos" else (255, 200, 0) if current_mode == "voz" else (100, 200, 255)
        
        put_text(canvas, status, 40, 60, scale=1.0, color=TEC_BLUE_LIGHT, thickness=2)
        put_text(canvas, f"Modo: {current_mode.upper()}", 40, 100, scale=0.9, color=mode_color, thickness=2)
        put_text(canvas, f"Senas: {current_sign_word}", 40, 140, scale=0.8)
        put_text(canvas, f"Voz: {current_voice_phrase}", 40, 180, scale=0.8)
        put_text(canvas, f"Teclado: {current_text_keyboard}", 40, 220, scale=0.8)

        put_text(canvas, "M=Mano | 'ayuda'/V=Voz | T=Teclado | Enter=Confirmar | Q/Esc=Salir",
                 40, 260, scale=0.6, color=(180,180,180), thickness=1)

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
            current_voice_phrase = ""
            add_log("SISTEMA", "Modo cambiado a VOZ (forzado por teclado)")
        elif key in (ord("t"), ord("T")):
            current_mode = "teclado"
            current_text_keyboard = ""
            add_log("SISTEMA", "Modo cambiado a TECLADO (escribe y Enter)")

        # Captura de texto en modo teclado
        if current_mode == "teclado":
            if 32 <= key <= 126:  # caracteres imprimibles básicos
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
                add_log("SENAS", f"Confirmada: {current_sign_word}")
                auto_translate(current_sign_word)
                current_sign_word = ""

            elif current_mode == "voz" and current_voice_phrase.strip():
                add_log("VOZ", f"Confirmada: {current_voice_phrase}")
                auto_translate(current_voice_phrase)
                current_voice_phrase = ""
                voice_active = False
                # Regresar a modo manos después de usar voz
                current_mode = "manos"
                add_log("SISTEMA", "Se regresa a modo MANOS")

            elif current_mode == "teclado" and current_text_keyboard.strip():
                add_log("TECLADO", f"Confirmada: {current_text_keyboard}")
                auto_translate(current_text_keyboard)
                current_text_keyboard = ""

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()