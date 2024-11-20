import cv2
import numpy as np
from ultralytics import YOLO
import math
import time
import random
import pygame

# Ініціалізація pygame для звуку
pygame.mixer.init()
shot_sound = pygame.mixer.Sound("D:/jammer/myGame/sound/blaster.mp3")

# Завантаження моделі
model = YOLO("D:/jammer/myGame/yolo8/yolov8n-drone.pt")
model.to('cuda')

# Підключення до камери
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not camera.isOpened():
    print("Не вдалося відкрити камеру")
    exit()

# Завантаження зображення дрона
drone_image_path = "D:/jammer/myGame/images/drone.png"
drone_image = cv2.imread(drone_image_path, cv2.IMREAD_UNCHANGED)

if drone_image is None:
    print("Не вдалося завантажити зображення дрона")
    exit()

# Розділення каналів (RGBA)
drone_rgb = drone_image[:, :, :3]
drone_alpha = drone_image[:, :, 3] if drone_image.shape[2] == 4 else np.ones_like(drone_image[:,:,0])
drone_alpha = cv2.normalize(drone_alpha, None, 0, 1, cv2.NORM_MINMAX)

# Масштаб дрона
drone_scale = 0.4
drone_h, drone_w = int(drone_rgb.shape[0] * drone_scale), int(drone_rgb.shape[1] * drone_scale)
drone_rgb = cv2.resize(drone_rgb, (drone_w, drone_h))
drone_alpha = cv2.resize(drone_alpha, (drone_w, drone_h))

# Глобальні змінні
auto_aim = False
score = 0
crosshair_x = 300
crosshair_y = 200
drone_active = True
frame_w, frame_h = 0, 0
explosion_effect = False
explosion_start_time = 0
explosion_duration = 0.3
explosion_pos = (0, 0)
drone_respawn_delay = 1.0
drone_respawn_time = 0
no_drone_period = False
current_accuracy = 0
accuracy_display_time = 0
accuracy_display_duration = 2.0

# Нові змінні для реалістичного автонаведення
auto_aim_noise = 5  # Похибка в пікселях
prediction_accuracy = 0.9  # Точність передбачення (90%)
model_confidence_threshold = 0.7  # Поріг впевненості моделі

class DroneMovement:
    def __init__(self):
        self.respawn()
        self.prev_x = self.x
        self.prev_y = self.y
        self.velocity_x = 0
        self.velocity_y = 0
        self.prediction_steps = 5
        self.last_predicted_pos = None
        self.prediction_error = 0

    def respawn(self):
        global no_drone_period, drone_respawn_time
        self.x = random.randint(drone_w, frame_w - drone_w) if frame_w > 0 else 300
        self.y = random.randint(drone_h, frame_h - drone_h) if frame_h > 0 else 200
        self.angle = random.uniform(0, 2 * math.pi)
        self.speed = random.uniform(3, 7)
        self.direction_change_time = time.time() + random.uniform(0.5, 2.0)
        no_drone_period = True
        drone_respawn_time = time.time() + drone_respawn_delay
    
    def update(self):
        global no_drone_period, drone_respawn_time
        current_time = time.time()
        
        if no_drone_period:
            if current_time >= drone_respawn_time:
                no_drone_period = False
            else:
                return None
        
        if current_time > self.direction_change_time:
            self.angle += random.uniform(-math.pi/4, math.pi/4)
            self.speed = random.uniform(3, 7)
            self.direction_change_time = current_time + random.uniform(0.5, 2.0)
        
        self.velocity_x = self.x - self.prev_x
        self.velocity_y = self.y - self.prev_y
        self.prev_x = self.x
        self.prev_y = self.y
        
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed
        
        if self.x < 0 or self.x > frame_w - drone_w:
            self.angle = math.pi - self.angle
            self.x = max(0, min(self.x, frame_w - drone_w))
        if self.y < 0 or self.y > frame_h - drone_h:
            self.angle = -self.angle
            self.y = max(0, min(self.y, frame_h - drone_h))
        
        # Оновлення похибки передбачення
        if self.last_predicted_pos is not None:
            pred_x, pred_y = self.last_predicted_pos
            actual_x, actual_y = int(self.x), int(self.y)
            self.prediction_error = math.sqrt((pred_x - actual_x)**2 + (pred_y - actual_y)**2)
        
        return int(self.x), int(self.y)

    def predict_position(self):
        if no_drone_period:
            return int(self.x), int(self.y)
        
        # Додаємо випадкову похибку до передбачення
        noise_x = random.uniform(-auto_aim_noise, auto_aim_noise)
        noise_y = random.uniform(-auto_aim_noise, auto_aim_noise)
        
        predicted_x = self.x + (self.velocity_x * self.prediction_steps * prediction_accuracy) + noise_x
        predicted_y = self.y + (self.velocity_y * self.prediction_steps * prediction_accuracy) + noise_y
        
        predicted_x = max(0, min(predicted_x, frame_w - drone_w))
        predicted_y = max(0, min(predicted_y, frame_h - drone_h))
        
        self.last_predicted_pos = (int(predicted_x), int(predicted_y))
        return self.last_predicted_pos

    def get_predicted_center(self):
        pred_x, pred_y = self.predict_position()
        return (pred_x + drone_w // 2, pred_y + drone_h // 2)

    def get_current_center(self):
        return (int(self.x + drone_w // 2), int(self.y + drone_h // 2))

def calculate_auto_aim_accuracy(detection_confidence, prediction_error):
    # Розрахунок точності на основі впевненості моделі та похибки передбачення
    base_accuracy = 85  # Базова точність автонаведення
    
    # Вплив впевненості моделі (до 10%)
    confidence_factor = (detection_confidence / model_confidence_threshold) * 10
    
    # Вплив похибки передбачення (до -15%)
    max_acceptable_error = drone_w / 2  # Максимально допустима похибка
    error_penalty = min(15, (prediction_error / max_acceptable_error) * 15)
    
    # Додавання випадкової варіації (±2%)
    random_variation = random.uniform(-2, 2)
    
    final_accuracy = base_accuracy + confidence_factor - error_penalty + random_variation
    return min(100, max(0, final_accuracy))

def calculate_shot_accuracy(shot_x, shot_y, current_center, predicted_center):
    current_distance = math.sqrt((shot_x - current_center[0])**2 + (shot_y - current_center[1])**2)
    predicted_distance = math.sqrt((shot_x - predicted_center[0])**2 + (shot_y - predicted_center[1])**2)
    max_distance = math.sqrt(drone_w**2 + drone_h**2) / 2
    
    if predicted_distance < current_distance:
        accuracy = max(0, 100 * (1 - predicted_distance / max_distance))
    else:
        accuracy = max(0, 70 * (1 - current_distance / max_distance))
    
    return min(100, accuracy)

def draw_crosshair(frame, x, y, size=20, color=(0, 0, 255)):
    cv2.line(frame, (x - size, y), (x + size, y), color, 2)
    cv2.line(frame, (x, y - size), (x, y + size), color, 2)
    cv2.circle(frame, (x, y), 2, color, -1)

def draw_explosion(frame, x, y, radius):
    overlay = frame.copy()
    cv2.circle(overlay, (x, y), radius, (0, 165, 255), -1)
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def mouse_callback(event, x, y, flags, param):
    global crosshair_x, crosshair_y, drone_active, score, explosion_effect
    global explosion_start_time, explosion_pos, current_accuracy, accuracy_display_time
    
    if not auto_aim:
        crosshair_x = x
        crosshair_y = y
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if no_drone_period:
            return
        
        shot_sound.play()
        
        current_center = drone_movement.get_current_center()
        predicted_center = drone_movement.get_predicted_center()
        
        if auto_aim:
            # Отримання даних від моделі YOLO
            results = model(frame)
            if len(results[0].boxes) > 0:
                box = results[0].boxes[0]
                detection_confidence = float(box.conf[0])
                
                # Розрахунок реалістичної точності автонаведення
                current_accuracy = calculate_auto_aim_accuracy(
                    detection_confidence,
                    drone_movement.prediction_error
                )
                
                if abs(crosshair_x - predicted_center[0]) < drone_w//2 and \
                   abs(crosshair_y - predicted_center[1]) < drone_h//2:
                    score += 1
                    explosion_effect = True
                    explosion_start_time = time.time()
                    explosion_pos = predicted_center
                    drone_movement.respawn()
        else:
            if abs(x - current_center[0]) < drone_w//2 and abs(y - current_center[1]) < drone_h//2:
                current_accuracy = calculate_shot_accuracy(x, y, current_center, predicted_center)
                score += 1
                explosion_effect = True
                explosion_start_time = time.time()
                explosion_pos = current_center
                drone_movement.respawn()
            else:
                current_accuracy = 0
        
        accuracy_display_time = time.time()

# Створення вікна та ініціалізація
cv2.namedWindow("Drone Hunter")
cv2.setMouseCallback("Drone Hunter", mouse_callback)
drone_movement = DroneMovement()

while True:
    ret, frame = camera.read()
    if not ret:
        break
        
    frame_h, frame_w = frame.shape[:2]
    
    if drone_active:
        updated_pos = drone_movement.update()
        
        if updated_pos is not None:
            x_pos, y_pos = updated_pos
            roi = frame[y_pos:y_pos + drone_h, x_pos:x_pos + drone_w]
            for c in range(3):
                roi[:, :, c] = roi[:, :, c] * (1 - drone_alpha) + drone_rgb[:, :, c] * drone_alpha
            frame[y_pos:y_pos + drone_h, x_pos:x_pos + drone_w] = roi
    
    if explosion_effect:
        current_time = time.time()
        if current_time - explosion_start_time < explosion_duration:
            progress = (current_time - explosion_start_time) / explosion_duration
            radius = int(50 * progress)
            draw_explosion(frame, explosion_pos[0], explosion_pos[1], radius)
        else:
            explosion_effect = False
    
    results = model(frame)
    
    if not no_drone_period:
        if auto_aim and drone_active and len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            if float(box.conf[0]) >= model_confidence_threshold:
                drone_predicted_x, drone_predicted_y = drone_movement.predict_position()
                crosshair_x = drone_predicted_x + drone_w // 2
                crosshair_y = drone_predicted_y + drone_h // 2
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    draw_crosshair(frame, crosshair_x, crosshair_y)

    # Відображення очків та статусу
    cv2.putText(frame, f'Score: {score}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    aim_status = "Auto-aim: ON" if auto_aim else "Auto-aim: OFF"
    cv2.putText(frame, aim_status, (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Відображення точності пострілу
    if time.time() - accuracy_display_time < accuracy_display_duration:
        accuracy_text = f"Accuracy: {current_accuracy:.1f}%"
        cv2.putText(frame, accuracy_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if no_drone_period:
        cv2.putText(frame, "RELOADING...", (frame_w//2 - 100, frame_h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.imshow("Drone Hunter", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord(' '):
        auto_aim = not auto_aim
        if not auto_aim:
            crosshair_x = frame_w // 2
            crosshair_y = frame_h // 2

camera.release()
cv2.destroyAllWindows()
pygame.quit()

