import subprocess
import time
import sys

print("\n" + "="*60)
print("🚀 SISTEMA COMPLETO DE ROBOT ASISTENTE TEC")
print("="*60 + "\n")

print("📋 Componentes del sistema:")
print("  1. Kinect AI (detección de obstáculos con U-Net)")
print("  2. Reconocimiento de señas y voz (webcam)")
print("  3. Sistema de navegación integrado")
print("  4. Comunicación MQTT optimizada")
print("\n")

# ====
# CONFIGURACIÓN DE RUTAS
# ====
# Ajusta estas rutas según tu sistema
PY_KINECT = r"C:\Users\OmarKings\Desktop\lidar\libfreenect-0.6.4\build\bin\Release\kinect_env_gpu\Scripts\python.exe"
KINECT_SCRIPT = r"C:\Users\OmarKings\Desktop\lidar\libfreenect-0.6.4\build\bin\Release\kinect_ai_collision\models\infer_real_time_mejorado.py"

PY_MAIN = r"C:\Users\OmarKings\Desktop\lidar\SignLanguage\venv_311\Scripts\python.exe"
MAIN_SCRIPT = r"C:\Users\OmarKings\Desktop\lidar\SignLanguage\main_mejorado.py"

# ====
# VERIFICAR ARCHIVOS
# ====
import os

print("🔍 Verificando archivos...")

if not os.path.exists(PY_KINECT):
    print(f"❌ ERROR: No se encuentra Python de Kinect: {PY_KINECT}")
    sys.exit(1)

if not os.path.exists(KINECT_SCRIPT):
    print(f"❌ ERROR: No se encuentra script de Kinect: {KINECT_SCRIPT}")
    print("   Asegurate de copiar 'infer_real_time_mejorado.py' a la carpeta correcta")
    sys.exit(1)

if not os.path.exists(PY_MAIN):
    print(f"❌ ERROR: No se encuentra Python principal: {PY_MAIN}")
    sys.exit(1)

if not os.path.exists(MAIN_SCRIPT):
    print(f"❌ ERROR: No se encuentra script principal: {MAIN_SCRIPT}")
    print("   Asegurate de copiar 'main_mejorado.py' a la carpeta correcta")
    sys.exit(1)

print("✅ Todos los archivos encontrados\n")

# ====
# INICIAR PROCESOS
# ====

print("="*60)
print("🟢 INICIANDO PROCESO KINECT (detección de obstáculos)")
print("="*60)
kinect_process = subprocess.Popen(
    [PY_KINECT, KINECT_SCRIPT],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

time.sleep(2)

print("\n" + "="*60)
print("🔵 INICIANDO PROCESO PRINCIPAL (señas, voz, navegación)")
print("="*60)
main_process = subprocess.Popen(
    [PY_MAIN, MAIN_SCRIPT],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

time.sleep(1)

# ====
# MONITOREO DEL SISTEMA
# ====

print("\n" + "="*60)
print("📡 SISTEMA EN EJECUCIÓN")
print("="*60)
print("\n🎮 CONTROLES:")
print("  • M = Modo manos (señas)")
print("  • V = Modo voz (o di 'ayuda')")
print("  • T = Modo teclado")
print("  • Enter = Confirmar frase")
print("  • Q/Esc = Salir")
print("\n🗺️  NAVEGACIÓN:")
print("  • Di o escribe: 'quiero ir a tims'")
print("  • Di o escribe: 'llevame a la biblioteca'")
print("  • Di o escribe: 'donde esta tim hortons'")
print("\n✋ CONTROL DE MOVIMIENTO (por señas):")
print("  • Deletrea 'MOVER' = Habilitar movimiento")
print("  • Deletrea 'PARAR' = Deshabilitar movimiento")
print("\n⚠️  CTRL + C para cerrar todo el sistema\n")
print("="*60 + "\n")

try:
    # Mantener el sistema corriendo
    while True:
        # Verificar si los procesos siguen vivos
        kinect_poll = kinect_process.poll()
        main_poll = main_process.poll()
        
        if kinect_poll is not None:
            print("\n⚠️  ADVERTENCIA: Proceso Kinect terminó inesperadamente")
            print(f"   Código de salida: {kinect_poll}")
            break
        
        if main_poll is not None:
            print("\n⚠️  ADVERTENCIA: Proceso principal terminó inesperadamente")
            print(f"   Código de salida: {main_poll}")
            break
        
        time.sleep(1)

except KeyboardInterrupt:
    print("\n\n" + "="*60)
    print("🛑 DETENIENDO SISTEMA...")
    print("="*60)
    
    print("\n🔴 Terminando proceso Kinect...")
    kinect_process.terminate()
    try:
        kinect_process.wait(timeout=5)
        print("✅ Proceso Kinect terminado")
    except subprocess.TimeoutExpired:
        print("⚠️  Forzando cierre de proceso Kinect...")
        kinect_process.kill()
    
    print("\n🔴 Terminando proceso principal...")
    main_process.terminate()
    try:
        main_process.wait(timeout=5)
        print("✅ Proceso principal terminado")
    except subprocess.TimeoutExpired:
        print("⚠️  Forzando cierre de proceso principal...")
        main_process.kill()
    
    print("\n" + "="*60)
    print("✔️  SISTEMA APAGADO CORRECTAMENTE")
    print("="*60 + "\n")

except Exception as e:
    print(f"\n❌ ERROR INESPERADO: {e}")
    print("\n🔴 Terminando procesos...")
    kinect_process.terminate()
    main_process.terminate()
    print("✔️  Procesos terminados\n")