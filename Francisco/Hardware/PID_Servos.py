from collections import deque
import sys
import time
from adafruit_servokit import ServoKit
import numpy as np

'''
Script de control PID versátil, elaborado originalmente 
para el control de la cámara y el seguimiento de objetos
con integración de servomotores (MG966R) usando la librería adafruit_servokit.
'''

# Configuración y constantes globales
NUM_SERVOS = 2  # Sistema tilt-Pan
PULSE_WIDTH_RANGE = (405, 2500)  # Ajustar de acuerdo a parámetros de servos
ACTUATION_RANGE = 180
LISTA_ANGULOS = [0, 90, 180, 90]  # Solamente son de prueba

# ============================================================
# Sistema PID
class PID:
    def __init__(self, kP, kI, kD):
        self.kP, self.kI, self.kD = kP, kI, kD
        self.last_error = 0
        self.integral = 0 
        self.last_time = time.time()

    def update(self, error):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0: 
            return 0

        # Proporcional 
        P = self.kP * error 

        # Integral (anti-windup)
        self.integral += error * dt
        self.integral = max(-10, min(10, self.integral))
        I = self.kI * self.integral
        
        # Derivativo
        D = self.kD * (error - self.last_error) / dt

        self.last_error = error
        self.last_time = now
        return P + I + D

# ============================================================
# Servomotores
def init_servos(num_servos: int = NUM_SERVOS, pwr: tuple = PULSE_WIDTH_RANGE, act_range: int = ACTUATION_RANGE) -> ServoKit:
    """Inicializa la placa PCA9685 y configura los rangos de pulso y grados de los servos."""
    try:
        kit = ServoKit(channels=16)
        
        for i in range(num_servos):
            kit.servo[i].set_pulse_width_range(pwr[0], pwr[1])
            kit.servo[i].actuation_range = act_range
            
        print(f"{num_servos} servos inicializados correctamente.")
        return kit
    except Exception as e:
        print(f"Error al inicializar los servomotores: {e}")
        return None

def mover_servo(servo, angle: int):
    """Mueve un servo específico a un ángulo validando los límites."""
    angle_restringido = max(0, min(angle, ACTUATION_RANGE))
    servo.angle = angle_restringido

# Función auxiliar para actualizar el servo de manera limpia
def actualizar_eje(error, dead_zone, pid_controller, current_angle, limits, servo_id):
    adjustment = 0.0
    if abs(error) > dead_zone:
        adjustment = pid_controller.update(error)
        current_angle = max(limits[0], min(limits[1], current_angle + adjustment))
        mover_servo(servo_id, current_angle)
    return current_angle, adjustment

def free_servos(kit, num_servos=NUM_SERVOS):
    """Lleva los servos a posición neutral (90°) de forma segura antes de salir."""
    if kit is not None:
        print("Llevando los servos a posición neutral (90°)...")
        try:
            for i in range(num_servos):
                kit.servo[i].angle = 90
        except Exception as e:
            print(f"No se pudo reposicionar los servos: {e}")

# ============================================================
# Ejemplo 
if __name__ == "__main__":
    # Inicializamos servos
    kit = init_servos()
    
    if kit is None:
        print(
            "Ningún servo pudo activarse, verificar:\n"
            "- Conexiones I2C\n"
            "- Conexiones Dupont / Alimentación externa"
        )
        sys.exit(1)

    # Asignación semántica clara dentro del flujo principal
    servo_x = kit.servo[0]  # Eje horizontal
    servo_y = kit.servo[1]  # Eje vertical

    print("Ambos servos activados.")

    try:
        # Secuencia basada en lista de ángulos
        for angle in LISTA_ANGULOS:
            mover_servo(servo_x, angle)  # Esto moverá el servo horizontal
            time.sleep(2)  # Descanso entre movimientos.
            
    except KeyboardInterrupt:
        print("\nSecuencia interrumpida por el usuario.")
        
    finally:
        # Posición de descanso segura al terminar
        free_servos(kit)
        print("Programa finalizado correctamente.")
