import os
import cv2
import numpy as np
import pyttsx3
from googletrans import Translator
import sounddevice as sd
import queue, json
from vosk import Model, KaldiRecognizer
import speech_recognition as sr


# ====================================================
# UTILIDAD PARA GUARDAR INFO EN UN ARCHIVO
# ====================================================
def guardar_info(texto):
    with open("registro_sistema.txt", "a", encoding="utf-8") as f:
        f.write(texto + "\n")
    print("📁 Guardado en registro_sistema.txt")


# ====================================================
# 1) CÁMARA CENTRADA (de camara_view.py)
# ====================================================
def camara_centrada():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("⚠️ No se pudo abrir la cámara.")
        return

    print("Presiona 'q' para salir.")

    screen_w = 1280
    screen_h = 720

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al leer frame.")
            break

        h, w, _ = frame.shape
        crop_size = int(min(h, w) * 0.8)
        start_x = w // 2 - crop_size // 2
        start_y = h // 2 - crop_size // 2
        cropped = frame[start_y:start_y+crop_size, start_x:start_x+crop_size]

        ratio = min(screen_w / crop_size, (screen_h - 100) / crop_size)
        new_w = int(crop_size * ratio)
        new_h = int(crop_size * ratio)
        resized = cv2.resize(cropped, (new_w, new_h))

        white_bar = np.ones((100, new_w, 3), dtype=np.uint8) * 255
        combined = np.vstack((resized, white_bar))

        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        y_offset = (screen_h - combined.shape[0]) // 2
        x_offset = (screen_w - combined.shape[1]) // 2

        canvas[y_offset:y_offset+combined.shape[0],
               x_offset:x_offset+combined.shape[1]] = combined

        cv2.imshow("Cámara centrada", canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ====================================================
# 2) TEXTO A VOZ (de texto_a_voz.py)
# ====================================================
def texto_a_voz():
    while True:
        texto = input("Escribe algo para decir (o 'salir'): ")
        if texto.lower() == "salir":
            break

        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        engine.say(texto)
        engine.runAndWait()

        guardar_info(f"[Texto a Voz] {texto}")


# ====================================================
# 3) TEXTO → TRADUCCIÓN → VOZ
# ====================================================
def texto_traducido_a_voz():
    traductor = Translator()
    idioma = input("Idioma destino (ej: en, fr, ja): ")

    while True:
        texto = input("Texto a traducir ('salir' para terminar): ")
        if texto.lower() == "salir":
            break

        traduccion = traductor.translate(texto, dest=idioma).text
        print("Traducción:", traduccion)

        engine = pyttsx3.init()
        engine.say(traduccion)
        engine.runAndWait()

        guardar_info(f"[Texto Traducido] {texto} → {traduccion}")


# ====================================================
# 4) VOZ A TEXTO (VOSK)
# ====================================================
def voz_a_texto():
    q = queue.Queue()
    model = Model(lang="es")
    rec = KaldiRecognizer(model, 16000)

    def callback(indata, frames, time, status):
        q.put(bytes(indata))

    print("Habla... (di 'salir' para terminar)")

    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                texto = result.get("text", "").lower()
                if texto:
                    print("Dijiste:", texto)
                    guardar_info(f"[Voz a Texto] {texto}")

                if "salir" in texto:
                    break


# ====================================================
# 5) VOZ → TRADUCCIÓN → VOZ
# ====================================================
def voz_traductor():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("Ajustando ruido...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)
    print("Listo.")

    traductor = Translator()
    idioma = input("Idioma destino (ej: en, fr, ja): ")

    while True:
        try:
            with mic as source:
                print("\nEscuchando...")
                audio = recognizer.listen(source, phrase_time_limit=5)

            texto = recognizer.recognize_google(audio, language="es-MX")
            print("Original:", texto)

            if "salir" in texto.lower():
                break

            traduccion = traductor.translate(texto, dest=idioma).text
            print("Traducción:", traduccion)

            engine = pyttsx3.init()
            engine.say(traduccion)
            engine.runAndWait()

            guardar_info(f"[Voz Traducida] {texto} → {traduccion}")

        except:
            print("No entendí, intenta de nuevo.")


# ====================================================
# MENÚ PRINCIPAL
# ====================================================
def main():
    while True:
        print("\n=== MENÚ DEL SISTEMA ===")
        print("1) Cámara centrada")
        print("2) Texto a voz")
        print("3) Texto traducido a voz")
        print("4) Voz a texto")
        print("5) Voz traducida")
        print("0) Salir")

        op = input("Selecciona: ")

        if op == "1": camara_centrada()
        elif op == "2": texto_a_voz()
        elif op == "3": texto_traducido_a_voz()
        elif op == "4": voz_a_texto()
        elif op == "5": voz_traductor()
        elif op == "0": break
        else:
            print("Opción no válida.")


if __name__ == "__main__":
    main()
