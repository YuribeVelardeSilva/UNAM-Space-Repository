'''
Script dedicado a tareas de operación de cámara, como abrir la cámara y capturar imágenes.
compatible con Picamera2 y cámaras USB. En raspberry y windows.
'''
# -------------- FUNCIONES DE CÁMARA --------------
from collections import deque
import json  # Por implementar para guardar configuraciones de cámara en un archivo JSON
import os
import platform
import time

import cv2

# ============================================================
# CONFIGURACIÓN
# ============================================================
CAMERA_INDEX = 0
FRAME_WIDTH = 640  # Dimensiones de la camara, ajustar según sea necesario
FRAME_HEIGHT = 480
FRAME_FPS = 30
buffer_seconds = 5  # Duración del buffer en segundos


def init_camera():
  """Inicializa Picamera2 si está disponible; de lo contrario, USB."""

  try:
    from picamera2 import Picamera2

    cam = Picamera2()

    config = cam.create_video_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
    )

    cam.configure(config)
    cam.start()

    print("Picamera2 inicializada.")
    return cam, "picamera2"

  except ImportError:
    pass

  if platform.system() == "Windows":
    """Estamos en Windows, así que usamos CAP_DSHOW para evitar el retraso de la cámara."""

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
  else:
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

  if not cap.isOpened():
    cap = cv2.VideoCapture(CAMERA_INDEX)

  if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara.")

  cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
  cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)

  print("Cámara USB inicializada.")

  return cap, "usb"


def capture_frame(camera, camera_type):
  """Captura un frame."""

  if camera_type == "picamera2":
    frame = camera.capture_array()

    # Picamera2 entrega RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return frame

  ret, frame = camera.read()

  if not ret:
    return None

  return frame


def release_camera(camera, camera_type):
  """Libera recursos."""

  if camera_type == "picamera2":
    camera.stop()
  else:
    camera.release()

  # Destruye todas las ventanas de OpenCV de forma segura al finalizar
  cv2.destroyAllWindows()


def save_frame(frame, file_name="Captura"):
  """Guarda un frame en un archivo con timestamp."""
  timestamp = time.strftime("%Y%m%d_%H%M%S")
  file_name = f"{file_name}_{timestamp}.jpg"
  cv2.imwrite(file_name, frame)
  return file_name


def save_video_buffered(
    buffer, file_name="Video_rec", dimensions=(FRAME_WIDTH, FRAME_HEIGHT), fps=30
):
  """Guarda un video en un archivo usando un buffer de frames con timestamp."""
  timestamp = time.strftime("%Y%m%d_%H%M%S")
  file_name = f"{file_name}_{timestamp}.mp4"
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  out = cv2.VideoWriter(file_name, fourcc, fps, dimensions)
  for f in buffer:
    out.write(f)
  out.release()
  return file_name


if __name__ == "__main__":

  # Prueba de la cámara
  camera, camera_type = init_camera()

  # Configuramos el buffer de frames de manera segura
  if camera_type == "usb":
    fps = int(camera.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or fps > 60:  # Si la cámara devuelve un valor erróneo o 0
      fps = FRAME_FPS
  else:
    fps = FRAME_FPS

  max_frames = fps * buffer_seconds
  buffer = deque(maxlen=max_frames)

  # Creamos la ventana una sola vez antes de entrar al bucle para evitar conflictos
  cv2.namedWindow("Cámara", cv2.WINDOW_NORMAL)

  try:
    while True:
      frame = capture_frame(camera, camera_type)

      if frame is None:
        print("No se pudo capturar el frame.")
        break

      # Añadimos el frame actual al buffer
      buffer.append(frame)

      cv2.imshow("Cámara", frame)

      key = cv2.waitKey(1) & 0xFF

      if key == ord("s"):
        print("Guardando video...")
        filename = save_video_buffered(
            buffer, "Video_rec", (FRAME_WIDTH, FRAME_HEIGHT), fps
        )
        print(f"Video guardado como {filename}")

      if key == ord("c"):
        print("Guardando imagen...")
        filename = save_frame(frame)
        print(f"Imagen guardada como {filename}")

      if key == ord("q"):
        break
  finally:
    # Aseguramos que la cámara y las ventanas se liberen incluso si ocurre un error forzado
    release_camera(camera, camera_type)
    print("Recursos liberados correctamente.")
