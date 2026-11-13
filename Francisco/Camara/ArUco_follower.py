'''
Script de Seguimiento de marcadores ArUco
Desarrollado originalmente para proyecto
UnamSpace, con la variante de control de camara
original de la Tesis de Francisco Nava
'''

# Librerías generales
import csv
import json
import time
import cv2
import cv2.aruco as aruco
import numpy as np
from pathlib import Path
from picamera2 import Picamera2
from dataclasses import dataclass

# Librerías para tareas de CV
from scr.Vision import Apply_cam_calib
from scr.Vision import Camara


# Librerías para tareas de hardware
from scr.Hardware import Motores_DC_correct
from scr.Hardware import MG966R_control as MG_servo
from scr.Hardware.PID_camera import PID

@dataclass
class MarkerPose:
    marker_id: int
    rvec: np.ndarray
    tvec: np.ndarray
    distance: float

# --- Rutas y configuración de archivos ---
RUTAS = {
    "Json_calib": Path(
        "/home/pacon/Tesis-Robot/scr/Vision/data_calib/raspberry_pi_hq_calibration.json"
    )
}

# --- Parámetros de la cámara y control ---
DEFAULT_K_AREA = 169889133.06
DEFAULT_B_AREA = -20.94
DEFAULT_FX = 909.897352
DEFAULT_W_REAL = 15.0  # Ancho físico real del marcador en centímetros
AREA_DETECT_TARGET = 1200

TARGET_AREA = 153000  # Área ideal que debe ocupar el objeto en píxeles
AREA_DEAD_ZONE = 0.05  # 5% de tolerancia
DEAD_ZONE = 5
DETECTION_TIMEOUT = 0.2

# --- Parámetros de servomotores y chasis ---
PINES_IZQ = (17, 27, 12)  # Forward, Backward, Enable (PWM)
PINES_DER = (23, 22, 13)  # Forward, Backward, Enable (PWM)
PIN_STBY = 24

PULSE_WIDTH_RANGE = (405, 2500)
START_ANGLE = 90.0
SERVO_DEAD_ZONE_GIRAR = 20.0  # Si pasa de 70° o 110°, el chasis gira
ANGLE_X_LIMIT = [45, 135]
ANGLE_Y_LIMIT = [50, 110]

MARKER_LENGTH = 0.10  # metros

FRAME_W, FRAME_H = 640, 480
CENTER_X, CENTER_Y = FRAME_W // 2, FRAME_H // 2

# Estado del seguimiento
TARGET_MARKER_ID = 2  # Cambia al ID que quieras seguir

last_detection = None
last_error_x = 0.0
last_error_y = 0.0

error_x_filtrado = 0.0
error_y_filtrado = 0.0

cx, cy = None, None

#Funciones extraidas del repositorio Unam-Space

# Creanos objeto detector de marcadores ArUco compatible con distintas versiones de OpenCV
def create_aruco_detector(dict_type = cv2.aruco.DICT_6X6_100):
    """Crea el detector ArUco compatible con distintas versiones."""

    dictionary = cv2.aruco.getPredefinedDictionary(dict_type)

    if hasattr(aruco, "ArucoDetector"):

        parameters = aruco.DetectorParameters()

        #Objeto detector de marcadores ArUco
        detector = cv2.aruco.ArucoDetector(dictionary,
                                           parameters)

        return detector, True

    parameters = cv2.aruco.DetectorParameters_create()

    return (dictionary, parameters), False


def detect_aruco_markers(frame, detector_info):
    """Detecta marcadores ArUco en un frame."""

    # Si la cámara entrega RGB, usa COLOR_RGB2GRAY.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detector, new_api = detector_info

    if new_api:
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        dictionary, parameters = detector
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray,
            dictionary,
            parameters=parameters
        )

    return corners, ids, rejected


 #Estimación de pose de marcadores ArUco

def estimate_marker_pose(corners, ids,marker_lenght,camera_matrix,dist_coeffs):
    """Estima la pose de los marcadores ArUco detectados en un frame."""
    half = marker_lenght / 2.0
    object_points = np.array([
        [-half,half,0],
        [half,half,0],
        [half,-half,0],
        [-half,-half,0]
    ], dtype=np.float32)

    poses = []

    for i,marker_corners in enumerate(corners):
        marker_id = int(ids[i][0])

        image_points = marker_corners[0].astype(np.float32)

        success,rvec,tvec = cv2.solvePnP(object_points,image_points,
                                         camera_matrix,dist_coeffs,
                                         flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not success:
            continue
        distance = float(np.linalg.norm(tvec))

        poses.append(MarkerPose(marker_id=marker_id,rvec=rvec,
                                tvec=tvec,distance=distance))
    return poses

#Orientación
def rotation_to_euler(rvec):
    """Convierte rvec a ángulos Euler en grados."""

    rotation_matrix, _ = cv2.Rodrigues(rvec)

    sy = np.sqrt(rotation_matrix[0, 0] ** 2 +
        rotation_matrix[1, 0] ** 2)

    singular = sy < 1e-6

    if not singular:

        pitch = np.arctan2(rotation_matrix[2, 1],
                           rotation_matrix[2, 2])

        roll = np.arctan2(-rotation_matrix[2, 0],sy)

        yaw = np.arctan2(rotation_matrix[1, 0],
                         rotation_matrix[0, 0])

    else:

        pitch = np.arctan2(-rotation_matrix[1, 2],
                           rotation_matrix[1, 1])

        roll = np.arctan2(-rotation_matrix[2, 0],sy)

        yaw = 0

    return np.degrees([pitch, roll, yaw])

#Dibujado
def draw_pose(frame,corners,ids,poses,
              marker_length,camera_matrix,
              dist_coeffs):
    """Dibuja marcador, distancia, orientación y ejes."""

    if ids is None:
        return frame

    axis_length = marker_length

    axis_points = np.array([
        [0, 0, 0],
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, axis_length]
    ], dtype=np.float32)

    for pose, marker_corners, marker_id in zip(poses,corners
                                               ,ids.flatten()):

        cv2.aruco.drawDetectedMarkers(frame,[marker_corners],
                                      np.array([[marker_id]]))

        img_points, _ = cv2.projectPoints(axis_points,pose.rvec,
            pose.tvec,camera_matrix,dist_coeffs)

        img_points = img_points.astype(int)

        origin = tuple(img_points[0].ravel())
        x_axis = tuple(img_points[1].ravel())
        y_axis = tuple(img_points[2].ravel())
        z_axis = tuple(img_points[3].ravel())

        # X = rojo
        cv2.line(frame,origin,x_axis,(0, 0, 255),3)

        # Y = verde
        cv2.line(frame,origin,y_axis,(0, 255, 0),3)

        # Z = azul
        cv2.line(frame,origin,z_axis,(255, 0, 0),3)

        pitch, roll, yaw = rotation_to_euler(pose.rvec)

        corner = marker_corners[0][0].astype(int)

        cv2.putText(frame,
            f"ID: {marker_id}",
            tuple(corner + [0, -50]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,(255, 255, 255),2)

        cv2.putText(frame,
            f"Dist: {pose.distance:.2f} m",
            tuple(corner + [0, -30]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,(255, 255, 255),2)

        cv2.putText(frame,f"P:{pitch:.0f} R:{roll:.0f} Y:{yaw:.0f}",
            tuple(corner + [0, -10]),
            cv2.FONT_HERSHEY_SIMPLEX,0.45,(0, 255, 255),1)

    return frame

def dibujar_centros(frame, corners, ids):
    if ids is None:
        return frame

    centros = calcular_centro_aruco(corners)

    for marcador_id, (cx, cy) in zip(ids.flatten(), centros):
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        cv2.putText(
            frame,
            f"ID:{marcador_id} Centro:({cx},{cy})",
            (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1
        )

    return frame

def calcular_centro_aruco(corners):
    """
    Calcula los centros de todos los marcadores detectados.

    Retorna una lista de tuplas:
    (cx, cy), una por marcador.
    """
    centros = []

    for corner in corners:
        puntos = corner.reshape(4, 2)

        cx = int(np.mean(puntos[:, 0]))
        cy = int(np.mean(puntos[:, 1]))

        centros.append((cx, cy))

    return centros

def obtener_centro_objetivo(corners, ids, target_id):
    """
    Retorna el centro (cx, cy) del marcador target_id.
    Si no se detecta, retorna None.
    """
    if ids is None:
        return None

    ids_flat = ids.flatten()

    for i, marker_id in enumerate(ids_flat):
        if marker_id == target_id:
            puntos = corners[i].reshape(4, 2)

            cx = int(np.mean(puntos[:, 0]))
            cy = int(np.mean(puntos[:, 1]))

            return cx, cy

    return None

#Funcion adicional no relacionada con space unam    
def filtrar_error(error_actual, error_nuevo, alpha=0.2):
    """Filtro exponencial aplicado al error para suavizar el movimiento de servos y motores."""
    return error_actual * (1 - alpha) + error_nuevo * alpha

try:
    try:
        # Inicializar dispositivos
        camara = Camara.init_cam()
        if camara is None:
            print("Error crítico: No se pudo inicializar la cámara.")
            raise RuntimeError("No pudo inicializar la camara, comprobar conexiones")
    except RuntimeError as error:
        print(f"Error en conexiòn: {error}")

    kit = MG_servo.init_servos(2, PULSE_WIDTH_RANGE, 180)
    servo_x = kit.servo[0]  # Horizontal
    servo_y = kit.servo[1]  # Vertical

    #Lectura de calibracion JSON
    
    # Cargar calibración de la cámara
    try:
        mtx, dist = Apply_cam_calib.cargar_calibracion(RUTAS["Json_calib"])
        print("Parámetros de calibración cargados exitosamente.")
    except Exception as e:
        print(f"Error cargando la calibración: {e}")

    #Configuramos e detector de marcadores ArUco
    detector_info = create_aruco_detector(aruco.DICT_6X6_100)


    # Inicializar controladores PID de servomotores
    pid_x = PID(kP=0.010, kI=0.01, kD=0.001)
    pid_y = PID(kP=0.010, kI=0.01, kD=0.001)

    #Posicionamiento inicial de los servos en un "angulo seguro"
    angle_x, angle_y = START_ANGLE, START_ANGLE

    while True:
        frame_raw = camara.capture_array()
        frame_raw = cv2.rotate(frame_raw,cv2.ROTATE_180)

        if frame_raw is None:
            print("Advertencia: frame nulo.")
            break

        # Trabajamos sobre una copia para dibujar sin modificar
        # el frame original.
        frame = frame_raw.copy()

        corners, ids, rejected = detect_aruco_markers(frame_raw,detector_info)

        # Dibujar todos los marcadores detectados.
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            frame = dibujar_centros(frame, corners, ids)

        # Buscar únicamente el marcador objetivo.
        centro = obtener_centro_objetivo(corners,ids,TARGET_MARKER_ID)

        if centro is not None:
            cx, cy = centro
            last_detection = time.monotonic()

            # Error entre el centro de la imagen y el marcador.
            error_x = CENTER_X - cx
            error_y = CENTER_Y - cy

            last_error_x = error_x
            last_error_y = error_y

            # Filtro exponencial.
            error_x_filtrado = filtrar_error(error_x_filtrado,error_x)
            error_y_filtrado = filtrar_error(error_y_filtrado,error_y)

            estado_tracking = 1

            # Actualizar servos.
            angle_x, adjust_x = MG_servo.actualizar_eje(error=error_x_filtrado,
                                                        dead_zone=DEAD_ZONE,
                                                        pid_controller=pid_x,
                                                        current_angle=angle_x,
                                                        limits=ANGLE_X_LIMIT,
                                                        servo_id=servo_x,)

            angle_y, adjust_y = MG_servo.actualizar_eje(error=error_y_filtrado,
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

        cv2.imshow("Seguimiento ArUco", frame)

        tecla = cv2.waitKey(10) & 0xFF

        if tecla in (13, ord("q")):
            break

except Exception as e:
    print(f"Algo no funciono como deberia {e}")

finally:
    camara.stop()
    cv2.destroyAllWindows()

      
