#!/bin/bash
set -e  # Detiene la ejecución si ocurre cualquier error

echo " Actualizando sistema..."
sudo apt update && sudo apt upgrade -y

echo " Instalando dependencias del sistema y bindings..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    libcamera-apps \
    python3-picamera2 \
    python3-opencv \
    python3-gpiozero \
    python3-lgpio \
    python3-numpy 

echo " Creando entorno virtual .UnamSpace..."
if [ -d ".UnamSpace" ]; then
    echo " .UnamSpace ya existe, omitiendo creación"
else
    python3 -m venv .UnamSpace --system-site-packages
fi

echo " Activando entorno..."
source .UnamSpace/bin/activate

echo " Actualizando pip en el entorno..."
pip install --upgrade pip setuptools wheel

echo " Instalando paquetes de Python (ML + Adafruit)..."
pip install \
    scikit-learn \
    joblib \
    pandas \
    Adafruit-Blinka \
    adafruit-circuitpython-servokit \
    adafruit-circuitpython-pca9685

echo " Verificando instalaciones..."
python3 -c "import cv2; print('OpenCV OK')"
python3 -c "from gpiozero import LED; print('GPIOZERO OK')"
python3 -c "import sklearn; print('Scikit-learn OK')"
python3 -c "import board; print('Blinka OK')"

# =========================================================
#             FINAL ELEGANTE - UNAM SPACE
# =========================================================
sleep 1
clear # Limpia la pantalla para un máximo impacto

cat << 'EOF'
     _   _ _   _    _    __  __   ____  ____   _    ____ _____ 
    | | | | \ | |  / \  |  \/  | / ___||  _ \ / \  / ___| ____|
    | | | |  \| | / _ \ | |\/| | \___ \| |_) / _ \| |   |  _|  
    | |_| | |\  |/ ___ \| |  | |  ___) |  __/ ___ \ |___| |___ 
     \___/|_| \_/_/   \_\_|  |_| |____/|_| /_/   \_\____|_____|

===================================================================
           ¡ENTORNO DE DESARROLLO INSTALADO CON ÉXITO!
===================================================================
 - No olvides activar la comunicación I2C en tu Raspberry Pi si planeas usar sensores o actuadores que dependan de este protocolo.
 -Para comenzar a trabajar en este espacio de la Facultad de Ingeniería, 
 ejecuta el siguiente comando en tu terminal:

   source .UnamSpace/bin/activate

===================================================================
EOF
