@echo off
chcp 65001 >nul
title COMPILAR LIBFREENECT PARA WINDOWS (KINECT 360)
color 0A

echo =========================================================
echo === COMPILANDO LIBFREENECT PARA WINDOWS (KINECT 360) ===
echo =========================================================

:: ==== CONFIGURACION DE RUTAS ====
set ROOT_DIR=C:\Users\OmarKings\Desktop\lidar
set LIBFREENECT_DIR=%ROOT_DIR%\libfreenect-0.6.4
set LIBUSB_DIR=%ROOT_DIR%\libusb-1.0.29
set PTHREADS_DIR=%ROOT_DIR%\Pre-built.2
set FREEGLUT_DIR=%ROOT_DIR%\freeglut
set BUILD_DIR=%LIBFREENECT_DIR%\build
set MSBUILD_EXE="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe"

:: ==== LIMPIAR CACHE ANTERIOR ====
echo.
echo ---- LIMPIANDO CACHE ANTERIOR ----
if exist "%BUILD_DIR%" rd /s /q "%BUILD_DIR%"
mkdir "%BUILD_DIR%"
cd /d "%BUILD_DIR%"

:: ==== CONFIGURAR CMAKE ====
echo.
echo ---- CONFIGURANDO PROYECTO CON CMAKE ----

cmake .. ^
 -G "Visual Studio 17 2022" ^
 -A x64 ^
 -DLIBUSB_1_INCLUDE_DIR="%LIBUSB_DIR%\include" ^
 -DLIBUSB_1_LIBRARY="%LIBUSB_DIR%\VS2022\MS64\static\libusb-1.0.lib" ^
 -DTHREADS_PTHREADS_INCLUDE_DIR="%PTHREADS_DIR%\include" ^
 -DTHREADS_PTHREADS_WIN32_LIBRARY="%PTHREADS_DIR%\lib\x64\pthreadVC2.lib" ^
 -DGLUT_INCLUDE_DIR="%FREEGLUT_DIR%\include" ^
 -DGLUT_glut_LIBRARY="%FREEGLUT_DIR%\lib\x64\freeglut.lib" ^
 -DCMAKE_POLICY_VERSION_MINIMUM=3.5

if %errorlevel% neq 0 (
    echo ───❌ Error en la configuracion con CMake.
    pause
    exit /b
)

echo.
echo ✅ CONFIGURACION COMPLETADA CORRECTAMENTE.

:: ==== COMPILAR ====
echo ---- INICIANDO COMPILACION ----
%MSBUILD_EXE% ALL_BUILD.vcxproj /p:Configuration=Release /m

if %errorlevel% neq 0 (
    echo ───❌ Error en la compilacion.
    pause
    exit /b
)

echo.
echo ======================================================
echo ✅ COMPILACION FINALIZADA CORRECTAMENTE
echo ======================================================

:: ==== COPIAR DEPENDENCIAS ====
echo.
echo ---- COPIANDO DEPENDENCIAS NECESARIAS ----

if exist "%LIBFREENECT_DIR%\build\lib\Release\freenect.dll" (
    copy "%LIBFREENECT_DIR%\build\lib\Release\freenect.dll" "%LIBFREENECT_DIR%\build\bin\Release" >nul
    echo ✅ Copiado freenect.dll
)
if exist "%LIBFREENECT_DIR%\build\lib\Release\freenect_sync.dll" (
    copy "%LIBFREENECT_DIR%\build\lib\Release\freenect_sync.dll" "%LIBFREENECT_DIR%\build\bin\Release" >nul
    echo ✅ Copiado freenect_sync.dll
)
if exist "%PTHREADS_DIR%\dll\x64\pthreadVC2.dll" (
    copy "%PTHREADS_DIR%\dll\x64\pthreadVC2.dll" "%LIBFREENECT_DIR%\build\bin\Release" >nul
    echo ✅ Copiado pthreadVC2.dll
)
if exist "%FREEGLUT_DIR%\bin\x64\freeglut.dll" (
    copy "%FREEGLUT_DIR%\bin\x64\freeglut.dll" "%LIBFREENECT_DIR%\build\bin\Release" >nul
    echo ✅ Copiado freeglut.dll
)

echo.
echo ======================================================
echo ✅ TODAS LAS DEPENDENCIAS COPIADAS CORRECTAMENTE
echo ======================================================

:: ==== PRUEBA RAPIDA ====
echo.
echo Probando ejecucion de freenect-camtest.exe ...
echo (Asegurate de tener el Kinect conectado y con su fuente de energia)
cd "%LIBFREENECT_DIR%\build\bin\Release"
echo.
pause
freenect-camtest.exe
