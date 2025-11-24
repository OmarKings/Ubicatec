"""
Sistema Completo de Robot Asistente TEC
Ejecuta ambos procesos con sus entornos virtuales correctos
VERSIÓN CORREGIDA - Sin errores de readline
"""

import subprocess
import sys
import os
import time
import signal
import atexit

# ====
# CONFIGURACIÓN DE RUTAS
# ====

# Entornos virtuales
VENV_KINECT = r"C:\Users\OmarKings\Desktop\lidar\libfreenect-0.6.4\build\bin\Release\kinect_env_gpu\Scripts\python.exe"
VENV_MAIN = r"C:\Users\OmarKings\Desktop\lidar\SignLanguage\venv_311\Scripts\python.exe"

# Scripts
SCRIPT_KINECT = r"C:\Users\OmarKings\Desktop\lidar\libfreenect-0.6.4\build\bin\Release\kinect_ai_collision\models\infer_real_time_mejorado.py"
SCRIPT_MAIN = r"C:\Users\OmarKings\Desktop\lidar\SignLanguage\main_mejorado.py"

# Directorios de trabajo
WORKDIR_KINECT = r"C:\Users\OmarKings\Desktop\lidar\libfreenect-0.6.4\build\bin\Release\kinect_ai_collision\models"
WORKDIR_MAIN = r"C:\Users\OmarKings\Desktop\lidar\SignLanguage"

# ====
# VERIFICACIÓN DE ARCHIVOS
# ====

archivos_requeridos = {
    "Python kinect_env_gpu": VENV_KINECT,
    "Python venv_311": VENV_MAIN,
    "infer_real_time.py": SCRIPT_KINECT,
    "main.py": SCRIPT_MAIN
}

print("=" * 70)
print("SISTEMA COMPLETO DE ROBOT ASISTENTE TEC")
print("=" * 70)
print("\nComponentes del sistema:")
print("  1. Kinect AI (deteccion de obstaculos con U-Net)")
print("  2. Reconocimiento de senas y voz (webcam)")
print("  3. Sistema de navegacion integrado")
print("  4. Comunicacion MQTT optimizada")
print()

# Verificar archivos
print("Verificando archivos...")
faltantes = []
for nombre, ruta in archivos_requeridos.items():
    if os.path.exists(ruta):
        print(f"  OK {nombre}")
    else:
        print(f"  X {nombre} NO encontrado")
        print(f"    Ruta: {ruta}")
        faltantes.append(nombre)

if faltantes:
    print(f"\nFaltan {len(faltantes)} archivo(s). No se puede continuar.")
    print("\nVerifica las rutas en la seccion de configuracion del script.")
    time.sleep(5)
    sys.exit(1)

print("\nTodos los archivos encontrados")

# ====
# PROCESOS
# ====
proceso_kinect = None
proceso_main = None
cleanup_done = False


def cleanup(signum=None, frame=None):
    """Limpia procesos al salir - SIN input() para evitar readline error"""
    global cleanup_done

    if cleanup_done:
        return

    cleanup_done = True

    print("\n\nCerrando sistema...")

    if proceso_kinect and proceso_kinect.poll() is None:
        print("  Cerrando proceso Kinect...")
        proceso_kinect.terminate()
        try:
            proceso_kinect.wait(timeout=5)
            print("  OK Proceso Kinect cerrado")
        except subprocess.TimeoutExpired:
            print("  Forzando cierre de proceso Kinect...")
            proceso_kinect.kill()
            proceso_kinect.wait()
            print("  OK Proceso Kinect forzado a cerrar")

    if proceso_main and proceso_main.poll() is None:
        print("  Cerrando proceso principal...")
        proceso_main.terminate()
        try:
            proceso_main.wait(timeout=5)
            print("  OK Proceso principal cerrado")
        except subprocess.TimeoutExpired:
            print("  Forzando cierre de proceso principal...")
            proceso_main.kill()
            proceso_main.wait()
            print("  OK Proceso principal forzado a cerrar")

    print("\nSistema cerrado correctamente")
    time.sleep(2)


# Registrar manejador de señales y atexit
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)
atexit.register(cleanup)

# ====
# INICIAR PROCESOS
# ====
try:
    print("\n" + "=" * 70)
    print("INICIANDO PROCESO KINECT (deteccion de obstaculos)")
    print("=" * 70)
    print(f"Entorno: kinect_env_gpu")
    print(f"Script: {SCRIPT_KINECT}")
    print(f"Directorio: {WORKDIR_KINECT}")
    print()

    # Iniciar proceso Kinect con su entorno
    proceso_kinect = subprocess.Popen(
        [VENV_KINECT, SCRIPT_KINECT],
        cwd=WORKDIR_KINECT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    # Esperar un poco para ver si inicia correctamente
    print("Esperando inicializacion del Kinect...")
    time.sleep(4)

    if proceso_kinect.poll() is not None:
        # El proceso terminó inmediatamente - hay un error
        stdout, stderr = proceso_kinect.communicate()
        print("\nERROR: Proceso Kinect termino inesperadamente")
        print(f"   Codigo de salida: {proceso_kinect.returncode}")
        print("\nSTDOUT:")
        print(stdout if stdout else "(vacio)")
        print("\nSTDERR:")
        print(stderr if stderr else "(vacio)")
        print("\nPosibles causas:")
        print("   - Kinect no esta conectado (verifica luz verde)")
        print("   - Falta el modelo U-Net en C:\\Users\\OmarKings\\Desktop\\weights\\best_unet.pth")
        print("   - Error en unet_model.py")
        print("   - Otro programa esta usando el Kinect")
        print("\nPara diagnosticar, ejecuta:")
        print(f"   cd {WORKDIR_KINECT}")
        print(f"   {VENV_KINECT} infer_real_time.py")
        time.sleep(5)
        sys.exit(1)

    print("OK Proceso Kinect iniciado correctamente")

    print("\n" + "=" * 70)
    print("INICIANDO PROCESO PRINCIPAL (senas, voz, navegacion)")
    print("=" * 70)
    print(f"Entorno: venv_311")
    print(f"Script: {SCRIPT_MAIN}")
    print(f"Directorio: {WORKDIR_MAIN}")
    print()

    # Iniciar proceso principal con su entorno
    proceso_main = subprocess.Popen(
        [VENV_MAIN, SCRIPT_MAIN],
        cwd=WORKDIR_MAIN
    )

    print("Esperando inicializacion del asistente...")
    time.sleep(3)

    if proceso_main.poll() is not None:
        print("\nERROR: Proceso principal termino inesperadamente")
        print(f"   Codigo de salida: {proceso_main.returncode}")
        print("\nPara diagnosticar, ejecuta:")
        print(f"   cd {WORKDIR_MAIN}")
        print(f"   {VENV_MAIN} main.py")
        cleanup()
        sys.exit(1)

    print("OK Proceso principal iniciado correctamente")

    print("\n" + "=" * 70)
    print("SISTEMA EN EJECUCION")
    print("=" * 70)
    print("\nCONTROLES:")
    print("  - M = Modo manos (senas)")
    print("  - V = Modo voz (o di 'ayuda')")
    print("  - T = Modo teclado")
    print("  - Enter = Confirmar frase")
    print("  - Q/Esc = Salir")
    print("\nCONTROL DE MOVIMIENTO:")
    print("  - Deletrea 'MOVER' = Habilitar movimiento")
    print("  - Deletrea 'PARAR' = Deshabilitar movimiento")
    print("\nNAVEGACION:")
    print("  - Di o escribe: 'quiero ir a tims'")
    print("  - Di o escribe: 'llevame a la biblioteca'")
    print("\nVENTANAS:")
    print("  - Ventana 1: Kinect AI (deteccion de obstaculos)")
    print("  - Ventana 2: Asistente TEC (senas/voz)")
    print("\nCTRL + C para cerrar todo el sistema")
    print("\n" + "=" * 70)
    print()

    # Monitorear procesos
    print("Monitoreando procesos...")
    print("(Este script seguira ejecutandose hasta que cierres las ventanas o presiones Ctrl+C)")
    print()

    while True:
        # Verificar si algún proceso terminó
        kinect_status = proceso_kinect.poll()
        main_status = proceso_main.poll()

        if kinect_status is not None:
            print("\nADVERTENCIA: Proceso Kinect termino inesperadamente")
            print(f"   Codigo de salida: {kinect_status}")

            # Leer salida de error si está disponible
            try:
                stderr = proceso_kinect.stderr.read()
                if stderr:
                    print("\nError del proceso Kinect:")
                    print(stderr[:500])  # Primeros 500 caracteres
            except:
                pass

            print("\nCerrando sistema completo...")
            cleanup()
            break

        if main_status is not None:
            print("\nProceso principal termino")
            print("   (Usuario cerro la ventana o presiono Q)")
            print("\nCerrando sistema completo...")
            cleanup()
            break

        # Esperar un poco antes de verificar de nuevo
        time.sleep(1)

except KeyboardInterrupt:
    print("\n\nInterrupcion detectada (Ctrl+C)")
    cleanup()

except Exception as e:
    print(f"\nERROR INESPERADO: {e}")
    import traceback
    traceback.print_exc()
    print("\nCerrando sistema...")
    cleanup()

finally:
    # Asegurar que cleanup se ejecute
    if not cleanup_done:
        cleanup()
