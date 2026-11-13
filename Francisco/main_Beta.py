'''Script para exhibiciones'''

import csv
import json
import os
import pathlib
import sys
import time
from collections import deque

import cv2
import cv2.aruco as aruco
import numpy as np

# VISION (Asegúrate de que la estructura de carpetas sea correcta)
from Camara import ArUco, Open_cam

# HARDWARE
from Hardware import PID_Servos as PID

# ==========================================
# 1. RUTAS Y CONFIGURACIÓN INICIAL
# ==========================================
BASE_DIR = pathlib.Path(__file__).resolve().parent
CAM_CONFIG_PATH = BASE_DIR / "Camara" / "Cam_config.json"
SERVO_CONFIG_PATH = BASE_DIR / "Data_calibs" / "Servos_config.json"

# Carga de configuración de la Cámara
try:
  with open(CAM_CONFIG_PATH, "r") as f:
    config_cam = json.load(f)
  CAMERA_INDEX = config_cam.get("CAMERA_INDEX", 0)
  FRAME_WIDTH = config_cam.get("FRAME_WIDTH", 640)
  FRAME_HEIGHT = config_cam.get("FRAME_HEIGHT", 480)
  FRAME_FPS = config_cam.get("FRAME_FPS", 30)
  BUFFER_SECONDS = config_cam.get("BUFFER_SECONDS", 5)
except FileNotFoundError:
  print(f"[AVISO] No se encontró {CAM_CONFIG_PATH}, usando valores por defecto.")
  FRAME_WIDTH, FRAME_HEIGHT, FRAME_FPS, BUFFER_SECONDS = 640, 480, 30, 5

CENTER_X, CENTER_Y = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
MARKER_LENGTH = 0.10  # Medida del marcador en metros
TARGET_MARKER = 2  # ID del marcador a seguir
DETECTION_TIMEOUT = 0.2
DEAD_ZONE = 10
MAX_HISTORIAL = 150

# Matriz de cámara y coeficientes (Reemplazar por calibración real si la tienes)
CAMERA_MATRIX = np.array(
    [[800.0, 0.0, FRAME_WIDTH / 2], [0.0, 800.0, FRAME_HEIGHT / 2], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
DIST_COEFFS = np.zeros((5, 1), dtype=np.float64)

# Carga de configuración de Servos
PULSE_WIDTH_RANGE = (405, 2500)  # Valores por defecto de respaldo
START_ANGLE = 90.0
SERVO_DEAD_ZONE_GIRAR = 20.0
ANGLE_X_LIMIT = [45, 135]
ANGLE_Y_LIMIT = [50, 110]

try:
  with open(SERVO_CONFIG_PATH, "r") as f:
    config_servos = json.load(f)
  PULSE_WIDTH_RANGE = tuple(config_servos["servo"]["pulse_width_range"])
  START_ANGLE = config_servos["servo"]["start_angle"]
  SERVO_DEAD_ZONE_GIRAR = config_servos["servo"]["servo_dead_zone_girar"]
  ANGLE_X_LIMIT = config_servos["limits"]["angle_x"]
  ANGLE_Y_LIMIT = config_servos["limits"]["angle_y"]
except FileNotFoundError:
  print(f"[AVISO] No se encontró {SERVO_CONFIG_PATH}, usando valores por defecto.")

# Inicializar controladores PID de servomotores
pid_x = PID(kP=0.010, kI=0.01, kD=0.001)
pid_y = PID(kP=0.010, kI=0.01, kD=0.001)

# ==========================================
# 2. INICIALIZACIÓN DE HARDWARE Y CÁMARA
# ==========================================
camara = None
cam_type = ""
kit = None

last_detection = None
last_error_x = 0.0
last_error_y = 0.0

error_x_filtrado = 0.0
error_y_filtrado = 0.0

#Funcion adicional no relacionada con space unam    
def filtrar_error(error_actual, error_nuevo, alpha=0.2):
    """Filtro exponencial aplicado al error para suavizar el movimiento de servos y motores."""
    return error_actual * (1 - alpha) + error_nuevo * alpha

try:
  # Inicializar Cámara 
  camara, cam_type = Open_cam.init_camera()
  if camara is None:
    raise RuntimeError("No se pudo inicializar la cámara (objeto es None).")

  fps = FRAME_FPS
  if cam_type == "usb":
    detected_fps = int(camara.get(cv2.CAP_PROP_FPS))
    if 0 < detected_fps <= 60:
      fps = detected_fps

  max_frames = fps * BUFFER_SECONDS
  buffer = deque(maxlen=max_frames)
  print(f"[OK] Cámara inicializada correctamente en modo: {cam_type}")

  # Inicializar Servos
  kit = PID.init_servos(2, PULSE_WIDTH_RANGE, 180)
  servo_x = kit.servo[0]  # Horizontal
  servo_y = kit.servo[1]  # Vertical

  PID.mover_servo(servo_x,START_ANGLE)
  PID.mover_servo(servo_y,START_ANGLE)
  print("Servos inicializados correctamente")

  # ==========================================
  # 3. BUCLE PRINCIPAL DE EXHIBICIÓN
  # ==========================================
  print("Todo bien. Iniciando bucle principal. Presiona 'q' para salir.")

  #Creaar el diccionario ArUco  que estaremos trabajando
  detector_info = ArUco.create_aruco_detector(aruco.DICT_6x6_100)

  while True:
    # Lectura específica para cámara USB (OpenCV VideoCapture)
    if cam_type == "usb":
      ret, frame = camara.read()
      if not ret or frame is None:
        print("[ERROR] No se pudo leer el fotograma de la cámara USB.")
        break
    else:
      # Por si en el futuro cambias a Picamera2
      frame = camara.capture_array()

    # Guardar frame en el búfer circular
    buffer.append(frame)


    # ---- Procesamiento de ARUCO y SERVOS ----
    corners,ids,rejected = ArUco.detect_aruco_markers(frame,detector_info)

    #Dibujar todos los marcadores detectados
    if ids is not None:
      cv2.aruco.drawDetectedMarkers(frame, corners, ids)
      frame = ArUco.dibujar_centros(frame, corners, ids)  

    #Buscar unicamente el marcador objetivo
    centro = ArUco.obtener_centro_objetivo(corners,ids,TARGET_MARKER)

    if centro is not None:
      cx,cy = centro
      last_detection = time.monotonic()

      #Error entre el centro de la imagen y el marcador

      error_x = CENTER_X - cx
      error_y = CENTER_Y - cy

      last_error_x = error_x
      last_error_y = error_y

      # Filtro exponencial.
      error_x_filtrado = filtrar_error(error_x_filtrado,error_x)
      error_y_filtrado = filtrar_error(error_y_filtrado,error_y)

      estado_tracking = 1

      # Actualizar servos.
      angle_x, adjust_x = PID.actualizar_eje(error=error_x_filtrado,
                                                              dead_zone=DEAD_ZONE,
                                                              pid_controller=pid_x,
                                                              current_angle=angle_x,
                                                              limits=ANGLE_X_LIMIT,
                                                              servo_id=servo_x,)
      
      angle_y, adjust_y = PID.actualizar_eje(error=error_y_filtrado,
                                                              dead_zone=DEAD_ZONE,
                                                              pid_controller=pid_y,
                                                              current_angle=angle_y,
                                                              limits=ANGLE_Y_LIMIT,
                                                              servo_id=servo_y,)
     # Información visual del seguimiento.
      cv2.circle(frame,(CENTER_X, CENTER_Y),5,(255, 0, 0),-1)

      cv2.line(frame,(CENTER_X, CENTER_Y),(cx, cy),(0, 255, 255),2)

      cv2.putText(frame,f"ID:{TARGET_MARKER_ID} Centro:({cx},{cy})",
                (10, 25),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0, 255, 0),2)

      cv2.putText(frame,f"Error X:{error_x} Y:{error_y}",
                (10, 50),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0, 255, 255),2) 

    else:
                estado_tracking = 0
    
                # No seguir indefinidamente una posición antigua.
                # Durante el timeout se mantienen los últimos errores.
                if (last_detection is not None and
                     time.monotonic() - last_detection < DETECTION_TIMEOUT):
                    
                    error_x = last_error_x
                    error_y = last_error_y
    
                    # No actualizar el PID sin una medición nueva.
                    cv2.putText(frame,"Perdida momentanea del marcador",
                                (10, 25),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 165, 255),2)
    
                else:
                    # El marcador lleva demasiado tiempo perdido.
                    # Se conserva la posición actual de los servos.
                    cx, cy = None, None
    
                    cv2.putText(frame,"Marcador no detectado",(10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 255),2)

                    
    # Mostrar la imagen en pantalla (Nota: requiere entorno gráfico en la Raspberry Pi o conexión SSH con X11 Forwarding)
    cv2.imshow("Exhibicion UNAM-Space", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Condición de salida: presionar la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

except Exception as e:
  print(f"[ERROR CRÍTICO] Ocurrió un error durante la ejecución: {e}")

finally:
  # ==========================================
  # 4. LIMPIEZA Y CIERRE SEGURO DE RECURSOS
  # ==========================================
  if camara is not None:
    if cam_type == "usb":
      camara.release()  # Libera la cámara USB de OpenCV
    else:
      camara.stop()  # Detiene Picamera2 si aplica
      
  cv2.destroyAllWindows()
  print("[INFO] Recursos liberados correctamente. ¡Hasta luego!")