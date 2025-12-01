"""
========================================
SISTEMA COMPLETO ROBOT LOOK-Y v3.1
Con GPS Fallback integrado
========================================

Ejecuta 3 procesos:
1. Kinect AI - Detección de obstáculos
2. GPS Fallback - Coordenadas desde computadora
3. Asistente Principal - Señas, voz y navegación

Autor: Sistema LOOK-Y
Versión: 3.1 - Corregido
"""

import subprocess
import sys
import os
import time
import signal

# ====================================
# CONFIGURACIÓN DE RUTAS
# ====================================

# Entornos virtuales
VENV_KINECT = r"C:\Users\OmarKings\Desktop\lidar\libfreenect-0.6.4\build\bin\Release\kinect_env_gpu\Scripts\python.exe"
VENV_MAIN = r"C:\Users\OmarKings\Desktop\lidar\SignLanguage\venv_311\Scripts\python.exe"

# Scripts
SCRIPT_KINECT = r"C:\Users\OmarKings\Desktop\lidar\libfreenect-0.6.4\build\bin\Release\kinect_ai_collision\models\infer_real_time_mejorado.py"
SCRIPT_GPS_FALLBACK = r"C:\Users\OmarKings\Desktop\lidar\gps_fallback.py"
SCRIPT_MAIN = r"C:\Users\OmarKings\Desktop\lidar\SignLanguage\main_mejorado.py"

# Directorios de trabajo
WORKDIR_KINECT = r"C:\Users\OmarKings\Desktop\lidar\libfreenect-0.6.4\build\bin\Release\kinect_ai_collision\models"
WORKDIR_GPS = r"C:\Users\OmarKings\Desktop\lidar"
WORKDIR_MAIN = r"C:\Users\OmarKings\Desktop\lidar\SignLanguage"

# ====================================
# VERIFICACIÓN DE ARCHIVOS
# ====================================

archivos_requeridos = {
    "Python kinect_env_gpu": VENV_KINECT,
    "Python venv_311": VENV_MAIN,
    "infer_real_time_mejorado.py": SCRIPT_KINECT,
    "main_mejorado.py": SCRIPT_MAIN,
    "gps_fallback.py": SCRIPT_GPS_FALLBACK
}

print("=" * 80)
print("🚀 SISTEMA COMPLETO ROBOT LOOK-Y v3.0")
print("=" * 80)
print()
print("📋 Componentes del sistema:")
print("  1. Kinect AI - Detección de obstáculos con U-Net")
print("  2. Reconocimiento de señas y voz - Control por gestos")
print("  3. GPS Fallback - Coordenadas desde computadora")
print("  4. Sistema de navegación autónoma")
print("  5. Comunicación MQTT optimizada")
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

# ====================================
# PROCESOS
# ====================================
proceso_kinect = None
proceso_gps_fallback = None
proceso_main = None

def cleanup(signum=None, frame=None):
    """Limpia procesos al salir"""
    print("\n\n🛑 Cerrando sistema...")
    
    if proceso_kinect and proceso_kinect.poll() is None:
        print("  Cerrando proceso Kinect AI...")
        proceso_kinect.terminate()
        try:
            proceso_kinect.wait(timeout=5)
            print("  ✓ Proceso Kinect AI cerrado")
        except subprocess.TimeoutExpired:
            print("  ⚠️  Forzando cierre de proceso Kinect...")
            proceso_kinect.kill()
            proceso_kinect.wait()
            print("  ✓ Proceso Kinect forzado a cerrar")
    
    if proceso_gps_fallback and proceso_gps_fallback.poll() is None:
        print("  Cerrando proceso GPS Fallback...")
        proceso_gps_fallback.terminate()
        try:
            proceso_gps_fallback.wait(timeout=5)
            print("  ✓ Proceso GPS Fallback cerrado")
        except subprocess.TimeoutExpired:
            print("  ⚠️  Forzando cierre de GPS Fallback...")
            proceso_gps_fallback.kill()
            proceso_gps_fallback.wait()
            print("  ✓ Proceso GPS Fallback forzado a cerrar")
    
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

# ====================================
# INICIAR PROCESOS
# ====================================
try:
    # ====================================
    # PROCESO 1: KINECT AI
    # ====================================
    print("\n" + "=" * 80)
    print("🟢 INICIANDO PROCESO 1/3: KINECT AI")
    print("=" * 80)
    print(f"Entorno: kinect_env_gpu")
    print(f"Script: {SCRIPT_KINECT}")
    print(f"Directorio: {WORKDIR_KINECT}")
    print()
    
    proceso_kinect = subprocess.Popen(
        [VENV_KINECT, SCRIPT_KINECT],
        cwd=WORKDIR_KINECT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    print("⏳ Esperando inicialización del Kinect...")
    time.sleep(3)
    
    if proceso_kinect.poll() is not None:
        stdout, stderr = proceso_kinect.communicate()
        print("\n❌ ERROR: Proceso Kinect terminó inesperadamente")
        print(f"   Código de salida: {proceso_kinect.returncode}")
        print("\n📄 STDOUT:")
        print(stdout if stdout else "(vacío)")
        print("\n📄 STDERR:")
        print(stderr if stderr else "(vacío)")
        input("\nPresiona Enter para salir...")
        sys.exit(1)
    
    print("✅ Proceso Kinect iniciado correctamente")
    
    # ====================================
    # PROCESO 2: GPS FALLBACK
    # ====================================
    print("\n" + "=" * 80)
    print("🟢 INICIANDO PROCESO 2/3: GPS FALLBACK")
    print("=" * 80)
    print(f"Entorno: venv_311")
    print(f"Script: {SCRIPT_GPS_FALLBACK}")
    print(f"Directorio: {WORKDIR_GPS}")
    print()
    
    # Iniciar GPS Fallback SIN stdin (no necesita entrada)
    proceso_gps_fallback = subprocess.Popen(
        [VENV_MAIN, SCRIPT_GPS_FALLBACK],
        cwd=WORKDIR_GPS,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    print("⏳ Esperando inicialización del GPS Fallback...")
    time.sleep(2)
    
    if proceso_gps_fallback.poll() is not None:
        stdout, stderr = proceso_gps_fallback.communicate()
        print("\n❌ ERROR: Proceso GPS Fallback terminó inesperadamente")
        print(f"   Código de salida: {proceso_gps_fallback.returncode}")
        print("\n📄 STDOUT:")
        print(stdout if stdout else "(vacío)")
        print("\n📄 STDERR:")
        print(stderr if stderr else "(vacío)")
        cleanup()
        sys.exit(1)
    
    print("✅ Proceso GPS Fallback iniciado correctamente")
    
    # ====================================
    # PROCESO 3: ASISTENTE PRINCIPAL
    # ====================================
    print("\n" + "=" * 80)
    print("🟢 INICIANDO PROCESO 3/3: ASISTENTE PRINCIPAL")
    print("=" * 80)
    print(f"Entorno: venv_311")
    print(f"Script: {SCRIPT_MAIN}")
    print(f"Directorio: {WORKDIR_MAIN}")
    print()
    
    proceso_main = subprocess.Popen(
        [VENV_MAIN, SCRIPT_MAIN],
        cwd=WORKDIR_MAIN
    )
    
    print("⏳ Esperando inicialización del asistente...")
    time.sleep(3)
    
    if proceso_main.poll() is not None:
        print("\n❌ ERROR: Proceso principal terminó inesperadamente")
        print(f"   Código de salida: {proceso_main.returncode}")
        cleanup()
        sys.exit(1)
    
    print("✅ Proceso principal iniciado correctamente")
    
    # ====================================
    # SISTEMA EN EJECUCIÓN
    # ====================================
    print("\n" + "=" * 80)
    print("📡 SISTEMA EN EJECUCIÓN - 3 PROCESOS ACTIVOS")
    print("=" * 80)
    print()
    print("🎮 CONTROLES:")
    print("  • M = Modo manos (señas)")
    print("  • V = Modo voz (o di 'ayuda')")
    print("  • T = Modo teclado")
    print("  • Enter = Confirmar frase")
    print("  • Q/Esc = Salir")
    print()
    print("🗺️  NAVEGACIÓN:")
    print("  • Di o escribe: 'quiero ir a tim hortons'")
    print("  • Di o escribe: 'llevame a la biblioteca'")
    print("  • Di o escribe: 'donde esta la fuente'")
    print()
    print("✋ CONTROL DE MOVIMIENTO (por señas):")
    print("  • Deletrea 'MOVER' = Habilitar movimiento")
    print("  • Deletrea 'PARAR' = Deshabilitar movimiento")
    print()
    print("🎥 VENTANAS:")
    print("  • Ventana 1: Kinect AI (detección de obstáculos)")
    print("  • Ventana 2: Asistente TEC (señas/voz)")
    print()
    print("📡 GPS FALLBACK:")
    print("  • El sistema envía coordenadas GPS al ESP32 automáticamente")
    print("  • Si el módulo GPS del ESP32 no funciona, usa estas coordenadas")
    print()
    print("⚠️  CTRL + C para cerrar todo el sistema")
    print("=" * 80)
    print()
    
    # ====================================
    # MONITOREO DE PROCESOS
    # ====================================
    print("🔄 Monitoreando procesos...")
    print("   (Este script seguirá ejecutándose hasta que cierres las ventanas o presiones Ctrl+C)")
    print()
    
    while True:
        # Verificar estado de procesos
        kinect_status = proceso_kinect.poll()
        gps_status = proceso_gps_fallback.poll()
        main_status = proceso_main.poll()
        
        if kinect_status is not None:
            print("\n⚠️  ADVERTENCIA: Proceso Kinect terminó inesperadamente")
            print(f"   Código de salida: {kinect_status}")
            try:
                stderr = proceso_kinect.stderr.read()
                if stderr:
                    print("\n📄 Error del proceso Kinect:")
                    print(stderr[:500])
            except:
                pass
            print("\n🛑 Cerrando sistema completo...")
            cleanup()
            break
        
        if gps_status is not None:
            print("\n⚠️  ADVERTENCIA: Proceso GPS Fallback terminó inesperadamente")
            print(f"   Código de salida: {gps_status}")
            try:
                stderr = proceso_gps_fallback.stderr.read()
                if stderr:
                    print("\n📄 Error del proceso GPS Fallback:")
                    print(stderr[:500])
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
        
        # Esperar antes de verificar de nuevo
        time.sleep(1)

except KeyboardInterrupt:
    print("\n\n⚠️  Interrupción detectada (Ctrl+C)")
    cleanup()

except Exception as e:
    print(f"\n❌ Error inesperado: {e}")
    import traceback
    traceback.print_exc()
    print("\n🛑 Cerrando sistema...")
    cleanup()