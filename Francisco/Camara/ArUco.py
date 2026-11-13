''' Script enfocado en el uso de Aruco para la detección de marcadores y la estimación de pose. '''
# Importamos herramientas de Open_cam.py para abrir la cámara y capturar imágenes
from Open_cam import init_camera, capture_frame, save_frame, save_video_buffered, release_camera
import cv2
import numpy as np
# En versiones recientes de OpenCV, aruco se importa directamente desde cv2
import cv2.aruco as aruco
from collections import deque
import json
import os
from dataclasses import dataclass

# ============================================================
# CONFIGURACIÓN
# ============================================================
config_path = r"Francisco\Camara\Cam_config.json"

# Carga segura con manejo de excepciones por si el archivo no existe
try:
    with open(config_path, "r") as f:
        config = json.load(f)
    CAMERA_INDEX = config.get("CAMERA_INDEX", 0)
    FRAME_WIDTH = config.get("FRAME_WIDTH", 640)
    FRAME_HEIGHT = config.get("FRAME_HEIGHT", 480)
    FRAME_FPS = config.get("FRAME_FPS", 30)
    BUFFER_SECONDS = config.get("BUFFER_SECONDS", 5)
except FileNotFoundError:
    print(f"No se encontró el archivo de configuración en {config_path}. Usando valores por defecto.")
    CAMERA_INDEX = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FRAME_FPS = 30
    BUFFER_SECONDS = 5

MARKER_LENGTH = 0.10  # metros

# IMPORTANTE: Estos valores son SOLO de ejemplo. Lo correcto es obtenerlos mediante calibración.
CAMERA_MATRIX = np.array([
    [800.0,   0.0, 640.0],
    [  0.0, 800.0, 480.0],
    [  0.0,   0.0,   1.0]
], dtype=np.float64)

DIST_COEFFS = np.zeros((5, 1), dtype=np.float64)
# ============================================================

@dataclass
class MarkerPose:
    marker_id: int
    rvec: np.ndarray
    tvec: np.ndarray
    distance: float

def create_aruco_detector(dict_type = cv2.aruco.DICT_6X6_100):
    """Crea el detector ArUco compatible con distintas versiones de OpenCV."""
    dictionary = cv2.aruco.getPredefinedDictionary(dict_type)

    if hasattr(cv2.aruco, "ArucoDetector"):
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        return detector, True

    parameters = cv2.aruco.DetectorParameters_create()
    return (dictionary, parameters), False


def detect_aruco_markers(frame, detector_info):
    """Detecta marcadores ArUco en un frame dado."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector, new_api = detector_info

    if new_api: 
        corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    else:
        dictionary, parameters = detector
        # Corrección: Cambiado 'id' por 'ids' para mantener consistencia
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
        
    return corners, ids, rejectedImgPoints


def estimate_marker_pose(corners, ids, marker_length, camera_matrix, dist_coeffs):
    """Estima la pose de los marcadores ArUco detectados en un frame."""
    half = marker_length / 2.0
    object_points = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0]
    ], dtype=np.float32)

    poses = []

    for i, marker_corners in enumerate(corners):
        marker_id = int(ids[i][0])
        image_points = marker_corners[0].astype(np.float32)

        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points,
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not success:
            continue
            
        distance = float(np.linalg.norm(tvec))
        poses.append(MarkerPose(marker_id=marker_id, rvec=rvec, tvec=tvec, distance=distance))
        
    return poses


def rotation_to_euler(rvec):
    """Convierte rvec a ángulos Euler en grados."""
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        roll = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        roll = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = 0

    return np.degrees([pitch, roll, yaw])


def draw_pose(frame, corners, ids, poses, marker_length, camera_matrix, dist_coeffs):
    """Dibuja marcador, distancia, orientación y ejes."""
    if ids is None or len(ids) == 0:
        return frame

    axis_length = marker_length
    axis_points = np.array([
        [0, 0, 0],
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, axis_length]
    ], dtype=np.float32)

    for pose, marker_corners, marker_id in zip(poses, corners, ids.flatten()):
        cv2.aruco.drawDetectedMarkers(frame, [marker_corners], np.array([[marker_id]]))

        img_points, _ = cv2.projectPoints(
            axis_points, pose.rvec, pose.tvec, camera_matrix, dist_coeffs
        )
        img_points = img_points.astype(int)

        origin = tuple(img_points[0].ravel())
        x_axis = tuple(img_points[1].ravel())
        y_axis = tuple(img_points[2].ravel())
        z_axis = tuple(img_points[3].ravel())

        # X = rojo, Y = verde, Z = azul
        cv2.line(frame, origin, x_axis, (0, 0, 255), 3)
        cv2.line(frame, origin, y_axis, (0, 255, 0), 3)
        cv2.line(frame, origin, z_axis, (255, 0, 0), 3)

        pitch, roll, yaw = rotation_to_euler(pose.rvec)
        corner = marker_corners[0][0].astype(int)

        cv2.putText(frame, f"ID: {marker_id}", tuple(corner + [0, -50]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"Dist: {pose.distance:.2f} m", tuple(corner + [0, -30]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"P:{pitch:.0f} R:{roll:.0f} Y:{yaw:.0f}", tuple(corner + [0, -10]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    return frame


def Aruco_center(frame, corners, ids):
    """Calcula y dibuja el centro de los marcadores detectados."""
    if ids is None:
        return frame, 0, 0
        
    for corner in corners:
        pts = corner[0]
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())

        cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
        cv2.putText(frame, f"Centro: {cx},{cy}", (cx + 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame, cx, cy
    return frame, 0, 0


if __name__ == "__main__":
    # Inicializamos la cámara
    camera, cam_type = init_camera()

    # Configuramos el buffer de frames de manera segura
    if cam_type == "usb":
        fps = int(camera.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or fps > 60:
            fps = FRAME_FPS
    else:
        fps = FRAME_FPS

    max_frames = fps * BUFFER_SECONDS
    buffer = deque(maxlen=max_frames)
    print(f"Cámara inicializada: {cam_type}")

    # Configuramos el detector de marcadores ArUco
    detector_info = create_aruco_detector(cv2.aruco.DICT_6X6_100)

    # Creamos la ventana de OpenCV una sola vez para evitar múltiples instancias
    cv2.namedWindow("Cámara ArUco", cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = capture_frame(camera, cam_type)
        
            if frame is None:
                print("No se pudo capturar el frame.")
                break

            # Guardamos el frame actual en el búfer circular para el vídeo
            buffer.append(frame.copy())

            # Detectamos marcadores ArUco en el frame
            corners, ids, rejectedImgPoints = detect_aruco_markers(frame, detector_info)
            
            # Calcular pose si se detectan marcadores
            if ids is not None and len(ids) > 0:
                poses = estimate_marker_pose(corners, ids, MARKER_LENGTH, CAMERA_MATRIX, DIST_COEFFS)
                frame = draw_pose(frame, corners, ids, poses, MARKER_LENGTH, CAMERA_MATRIX, DIST_COEFFS)
                frame, _, _ = Aruco_center(frame, corners, ids)
                
            cv2.imshow("Cámara ArUco", frame)
            
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                print("Guardando video...")
                filename = save_video_buffered(buffer, "Video_rec", (FRAME_WIDTH, FRAME_HEIGHT), fps)
                print(f"Video guardado como {filename}")

            if key == ord('c'):
                print("Guardando imagen...")
                filename = save_frame(frame)
                print(f"Imagen guardada como {filename}")

            if key == ord('q'):
                break
    finally:
        release_camera(camera, cam_type)
        print("Recursos liberados correctamente.")
