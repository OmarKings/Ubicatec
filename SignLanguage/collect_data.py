import cv2
import mediapipe as mp
import os
import numpy as np
import time  # Added for save delay

# Carpeta donde guardar los datos
DATA_DIR = './data'
os.makedirs(DATA_DIR, exist_ok=True)

# Inicializa MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Keep at 1 to avoid confusion
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5  # Better for video tracking stability
)
mp_draw = mp.solutions.drawing_utils

# Configura cámara con error handling
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(" Error: No se pudo acceder a la cámara. Verifica permisos o conexión.")
    exit(1)

# FIX: Validación para cualquier signo/palabra (no solo 1 letra)
while True:
    sign_input = input("¿Qué signo/palabra vas a capturar (e.g., A, COMER, BEBER)? ").upper().strip()
    if len(sign_input) > 0 and sign_input.replace('Ñ', '').isalpha():  # Permite letras + Ñ, pero cualquier longitud
        sign = sign_input
        break
    else:
        print(" Entrada inválida. Usa solo letras (A-Z, Ñ) sin números/espacios.")

# Directorio de guardado
save_dir = os.path.join(DATA_DIR, sign)
os.makedirs(save_dir, exist_ok=True)

# FIX: Cuenta archivos existentes para continuar si folder ya existe
existing_files = [f for f in os.listdir(save_dir) if f.endswith('.npy')]
existing_count = len(existing_files)
print(f" Folder '{save_dir}' ya existe con {existing_count} muestras.")

# Pedir número de muestras deseadas (total, incluyendo existentes)
try:
    target_count = int(input(f"¿Cuántas muestras TOTALES para '{sign}'? (Recomendado: 100-200) "))
    if target_count <= existing_count:
        print(f"  Ya tienes {existing_count} muestras. Ajusta a más si quieres agregar.")
        target_count = max(target_count, existing_count + 1)
except ValueError:
    print(" Número inválido. Usando 100 totales.")
    target_count = 100

to_collect = target_count - existing_count  # Solo recolecta lo que falta
print(f"Capturando {to_collect} nuevas muestras para '{sign}' (total: {target_count}).")
print("Instrucciones: Realiza el signo completo con mano derecha. Varía ligeramente (ángulos, velocidad).")
print("Presiona 's' para guardar manualmente (ideal para movimientos), 'q' o ESC para salir.")

count = existing_count  # Empieza desde el existente
last_save_time = time.time()  # Para delay entre saves
save_delay = 1.0  # Segundos entre saves automáticos (aumentado para signos dinámicos)

# FIX: Opciones de mano y flip
USE_RIGHT_HAND_ONLY = True  # True = solo derecha (estándar LSM); False = ambas
ENABLE_FLIP = True  # True = vista mirror (natural)

hand_label_text = "derecha" if USE_RIGHT_HAND_ONLY else "cualquiera"

print(f"  Usando mano {hand_label_text} (flip: {'on' if ENABLE_FLIP else 'off'}).")

while count < target_count:
    ret, frame = cap.read()
    if not ret:
        print(" Error al leer frame de la cámara.")
        break

    # FIX: Flip horizontal para vista mirror (opcional)
    if ENABLE_FLIP:
        frame = cv2.flip(frame, 1)

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)

    if results.multi_hand_landmarks:
        for handLms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            
            # FIX: Ajuste handedness por flip (invierte si mirror)
            if ENABLE_FLIP:
                label = 'Right' if label == 'Left' else 'Left'
            
            # FIX: Filtro corregido para mano derecha only
            if USE_RIGHT_HAND_ONLY and label != 'Left':
                continue  # Salta si no es derecha (post-ajuste)

            lmList = np.array([[lm.x, lm.y] for lm in handLms.landmark])

            # FIX: Mirror landmarks si es left pero queremos simular right (opcional, para consistencia)
            # (Desactívalo si recolectas directamente con right)
            if not USE_RIGHT_HAND_ONLY and label == 'Left':
                lmList[:, 0] = 1.0 - lmList[:, 0]  # Flip x para match right-pose

            # Normaliza respecto a la muñeca (primer landmark)
            lmList -= lmList[0]

            # Escala a -1 / 1
            max_val = np.max(np.abs(lmList))
            if max_val != 0:
                lmList /= max_val

            # Aplanar
            lmList_flat = lmList.flatten()

            # Guardar con delay o manual
            current_time = time.time()
            key = cv2.waitKey(1) & 0xFF
            save_condition = (current_time - last_save_time > save_delay) or (key == ord('s'))
            if save_condition and count < target_count:
                try:
                    np.save(os.path.join(save_dir, f"{count:04d}.npy"), lmList_flat)  # Padding para orden
                    count += 1
                    last_save_time = current_time
                    print(f" Guardada muestra {count}/{target_count} para '{sign}'")
                except Exception as e:
                    print(f" Error al guardar: {e}")

            # Dibujar landmarks y handedness
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, f'Mano: {label}', (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Overlay de info
    remaining = target_count - count
    cv2.putText(frame, f'Signo: {sign} | Muestras: {count}/{target_count} (faltan: {remaining})', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "'s': Guardar manual | 'q'/ESC: Salir | Mantén signo claro", (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow(f'Recolectando datos para "{sign}" - Mano {hand_label_text}', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

# Limpieza
cap.release()
cv2.destroyAllWindows()
total_saved = count  # Incluye existentes
print(f" Sesión terminada. {total_saved} muestras totales en {save_dir}.")
if count < target_count:
    print(f"  Meta no alcanzada (faltan {target_count - count}). Ejecuta de nuevo.")
else:
    print(" ¡Meta alcanzada! Ahora puedes entrenar el modelo con el script de training.")
