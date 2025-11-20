import subprocess
import time

print("\n🚀 Iniciando sistema con DOS dispositivos...\n")

# ============================================================
# 1) PROCESO KINECT  (U-Net + libfreenect)
# ============================================================

PY_KINECT = r"C:\Users\OmarKings\Desktop\lidar\libfreenect-0.6.4\build\bin\Release\kinect_env_gpu\Scripts\python.exe"
KINECT_SCRIPT = r"C:\Users\OmarKings\Desktop\lidar\libfreenect-0.6.4\build\bin\Release\kinect_ai_collision\models\infer_real_time.py"

print("🟢 Iniciando proceso Kinect...")
kinect_process = subprocess.Popen([PY_KINECT, KINECT_SCRIPT])

time.sleep(1)


# ============================================================
# 2) PROCESO SEÑAS  (mediapipe + sklearn + webcam)
# ============================================================

PY_SIGN = r"C:\Users\OmarKings\Desktop\lidar\SignLanguage\venv_311\Scripts\python.exe"
SIGN_SCRIPT = r"C:\Users\OmarKings\Desktop\lidar\SignLanguage\main.py"

print("🔵 Iniciando proceso de SEÑAS (webcam)...")
sign_process = subprocess.Popen([PY_SIGN, SIGN_SCRIPT])


# ============================================================
# 3) LOOP PRINCIPAL
# ============================================================

print("\n📡 Ambos procesos ejecutándose.\nCTRL + C para cerrar todo.\n")

try:
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("\n🛑 Terminando procesos...")
    kinect_process.terminate()
    sign_process.terminate()
    print("✔ Sistema apagado.\n")
