import time
import os

FILE_PATH = "current_gesture.txt"

print("Leyendo seña actual en tiempo real (Ctrl+C para salir)...")

last_gesture = None

try:
    while True:
        if os.path.exists(FILE_PATH):
            with open(FILE_PATH, "r", encoding="utf-8") as f:
                gesture = f.read().strip()
            if gesture != last_gesture:
                print("Seña detectada:", gesture)
                last_gesture = gesture
        time.sleep(0.1)  # 100 ms, evita sobrecargar la CPU
except KeyboardInterrupt:
    print("Programa terminado por usuario.")
