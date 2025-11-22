"""
Sistema Completo de Robot Asistente TEC
Ejecuta ambos procesos con sus entornos virtuales correctos
Adaptado a la estructura real del proyecto
"""

import subprocess
import sys
import os
import time
import signal

# ====================================================================
# 🔧 CONFIGURACIÓN DE RUTAS - ADAPTADO A TU ESTRUCTURA REAL
# ====================================================================

# Entornos virtuales
VENV_KINECT = r"C:\Users\OmarKings\Desktop\lidar\libfreenect-0.6.4\build\bin\Release\kinect_env_gpu\Scripts\python.exe"
VENV_MAIN = r"C:\Users\OmarKings\Desktop\lidar\SignLanguage\venv_311\Scripts\python.exe"

# Scripts
SCRIPT_KINECT = r"C:\Users\OmarKings\Desktop\lidar\libfreenect-0.6.4\build\bin\Release\kinect_ai_collision\models\infer_real_time_mejorado.py"
SCRIPT_MAIN = r"C:\Users\OmarKings\Desktop\lidar\SignLanguage\main_mejorado.py"

# Directorios de trabajo
WORKDIR_KINECT = r"C:\Users\OmarKings\Desktop\lidar\libfreenect-0.6.4\build\bin\Release\kinect_ai_collision\models"
WORKDIR_MAIN = r"C:\Users\OmarKings\Desktop\lidar\SignLanguage"

# ====================================================================
# 📋 VERIFICACIÓN DE ARCHIVOS
# ====================================================================

archivos_requeridos = {
    "Python kinect_env_gpu": VENV_KINECT,
    "Python venv_311": VENV_MAIN,
    "infer_real_time.py": SCRIPT_KINECT,
    "main.py": SCRIPT_MAIN
}

print("=" * 70)
print("🚀 SISTEMA COMPLETO DE ROBOT ASISTENTE TEC")
print("=" * 70)
print("\n📋 Componentes del sistema:")
print("  1. Kinect AI (detección de obstáculos con U-Net)")
print("  2. Reconocimiento de señas y voz (webcam)")
print("  3. Sistema de navegación integrado")
print("  4. Comunicación MQTT optimizada")
print()

# Verificar archivos
print("🔍 Verificando archivos...")
faltantes = []
for nombre, ruta in archivos_requeridos.items():
    if os.path.exists(ruta):
        print(f"  ✓ {nombre}")
    else:
        print(f"  ✗ {nombre} NO encontrado")
        print(f"    Ruta: {ruta}")
        faltantes.append(nombre)

if faltantes:
    print(f"\n❌ Faltan {len(faltantes)} archivo(s). No se puede continuar.")
    print("\n💡 Verifica las rutas en la sección de configuración del script.")
    input("\nPresiona Enter para salir...")
    sys.exit(1)

print("\n✅ Todos los archivos encontrados")

# ====================================================================
# 🔧 PROCESOS
# ====================================================================
proceso_kinect = None
proceso_main = None

def cleanup(signum=None, frame=None):
    """Limpia procesos al salir"""
    print("\n\n🛑 Cerrando sistema...")
    
    if proceso_kinect and proceso_kinect.poll() is None:
        print("  Cerrando proceso Kinect...")
        proceso_kinect.terminate()
        try:
            proceso_kinect.wait(timeout=5)
            print("  ✓ Proceso Kinect cerrado")
        except subprocess.TimeoutExpired:
            print("  ⚠️  Forzando cierre de proceso Kinect...")
            proceso_kinect.kill()
            proceso_kinect.wait()
            print("  ✓ Proceso Kinect forzado a cerrar")
    
    if proceso_main and proceso_main.poll() is None:
        print("  Cerrando proceso principal...")
        proceso_main.terminate()
        try:
            proceso_main.wait(timeout=5)
            print("  ✓ Proceso principal cerrado")
        except subprocess.TimeoutExpired:
            print("  ⚠️  Forzando cierre de proceso principal...")
            proceso_main.kill()
            proceso_main.wait()
            print("  ✓ Proceso principal forzado a cerrar")
    
    print("\n✅ Sistema cerrado correctamente")
    print("\nPresiona Enter para salir...")
    input()
    sys.exit(0)

# Registrar manejador de señales
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

# ====================================================================
# 🚀 INICIAR PROCESOS
# ====================================================================
try:
    print("\n" + "=" * 70)
    print("🟢 INICIANDO PROCESO KINECT (detección de obstáculos)")
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
    print("⏳ Esperando inicialización del Kinect...")
    time.sleep(4)
    
    if proceso_kinect.poll() is not None:
        # El proceso terminó inmediatamente - hay un error
        stdout, stderr = proceso_kinect.communicate()
        print("\n❌ ERROR: Proceso Kinect terminó inesperadamente")
        print(f"   Código de salida: {proceso_kinect.returncode}")
        print("\n📄 STDOUT:")
        print(stdout if stdout else "(vacío)")
        print("\n📄 STDERR:")
        print(stderr if stderr else "(vacío)")
        print("\n💡 Posibles causas:")
        print("   • Kinect no está conectado (verifica luz verde)")
        print("   • Falta el modelo U-Net en C:\\Users\\OmarKings\\Desktop\\weights\\best_unet.pth")
        print("   • Error en unet_model.py")
        print("   • Otro programa está usando el Kinect")
        print("\n🔧 Para diagnosticar, ejecuta:")
        print(f"   cd {WORKDIR_KINECT}")
        print(f"   {VENV_KINECT} infer_real_time.py")
        input("\nPresiona Enter para salir...")
        sys.exit(1)
    
    print("✅ Proceso Kinect iniciado correctamente")
    
    print("\n" + "=" * 70)
    print("🔵 INICIANDO PROCESO PRINCIPAL (señas, voz, navegación)")
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
    
    print("⏳ Esperando inicialización del asistente...")
    time.sleep(3)
    
    if proceso_main.poll() is not None:
        print("\n❌ ERROR: Proceso principal terminó inesperadamente")
        print(f"   Código de salida: {proceso_main.returncode}")
        print("\n🔧 Para diagnosticar, ejecuta:")
        print(f"   cd {WORKDIR_MAIN}")
        print(f"   {VENV_MAIN} main.py")
        cleanup()
        sys.exit(1)
    
    print("✅ Proceso principal iniciado correctamente")
    
    print("\n" + "=" * 70)
    print("📡 SISTEMA EN EJECUCIÓN")
    print("=" * 70)
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
    print("\n🎥 VENTANAS:")
    print("  • Ventana 1: Kinect AI (detección de obstáculos)")
    print("  • Ventana 2: Asistente TEC (señas/voz)")
    print("\n⚠️  CTRL + C para cerrar todo el sistema")
    print("\n" + "=" * 70)
    print()
    
    # Monitorear procesos
    print("🔄 Monitoreando procesos...")
    print("   (Este script seguirá ejecutándose hasta que cierres las ventanas o presiones Ctrl+C)")
    print()
    
    while True:
        # Verificar si algún proceso terminó
        kinect_status = proceso_kinect.poll()
        main_status = proceso_main.poll()
        
        if kinect_status is not None:
            print("\n⚠️  ADVERTENCIA: Proceso Kinect terminó inesperadamente")
            print(f"   Código de salida: {kinect_status}")
            
            # Leer salida de error si está disponible
            try:
                stderr = proceso_kinect.stderr.read()
                if stderr:
                    print("\n📄 Error del proceso Kinect:")
                    print(stderr[:500])  # Primeros 500 caracteres
            except:
                pass
            
            print("\n🛑 Cerrando sistema completo...")
            cleanup()
            break
        
        if main_status is not None:
            print("\n✅ Proceso principal terminó")
            print("   (Usuario cerró la ventana o presionó Q)")
            print("\n🛑 Cerrando sistema completo...")
            cleanup()
            break
        
        # Esperar un poco antes de verificar de nuevo
        time.sleep(1)

except KeyboardInterrupt:
    print("\n\n⚠️  Interrupción detectada (Ctrl+C)")
    cleanup()

except Exception as e:
    print(f"\n❌ ERROR INESPERADO: {e}")
    import traceback
    traceback.print_exc()
    print("\n🛑 Cerrando sistema...")
    cleanup()