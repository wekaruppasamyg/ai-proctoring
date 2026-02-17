import cv2
import mediapipe as mp
import math
import random
import time
import webbrowser
import os

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

angle = 0

# ================= STATE =================
menu_active = False
login_opened = False
particle = None
hover_start = None
HOLD_TIME = 2  # seconds

# ================= COLORS (IRON MAN UI) =================
CYAN = (255, 255, 0)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)
DIM_RED = (0, 0, 120)

# ================= 3D LOGIN PARTICLE =================
class LoginParticle3D:
    def __init__(self, cx, cy):
        self.base_x = cx
        self.base_y = cy - 120
        self.z = random.uniform(0.6, 1.4)
        self.z_speed = random.uniform(-0.003, 0.003)
        self.scale = 1 / self.z
        self.x = int(self.base_x)
        self.y = int(self.base_y)
        self.r = int(28 * self.scale)

    def update(self):
        self.z += self.z_speed
        if self.z < 0.5 or self.z > 1.5:
            self.z_speed *= -1
        self.scale = 1 / self.z
        self.x = int(self.base_x)
        self.y = int(self.base_y)
        self.r = int(28 * self.scale)

    def draw(self, frame, color, occluded=False):
        if not occluded:
            for i in range(14, 2, -3):
                cv2.circle(frame, (self.x, self.y),
                           self.r + i, color, 1)
            cv2.circle(frame, (self.x, self.y),
                       self.r, color, -1)
        else:
            cv2.circle(frame, (self.x, self.y),
                       self.r, DIM_RED, -1)

        cv2.putText(frame, "LOGIN",
                    (self.x - 38, self.y + self.r + 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85 * self.scale,
                    color, 2)

    def is_hovered(self, fx, fy):
        return math.hypot(self.x - fx, self.y - fy) < self.r

# ================= GESTURE =================
def is_open_palm(lm):
    tips = [8, 12, 16, 20]
    return sum(lm[t].y < lm[t - 2].y for t in tips) >= 4

# ================= HOLOGRAM UI =================
def draw_hologram_ring(frame, cx, cy, radius, angle_offset):
    for i in range(0, 360, 12):
        rad = math.radians(i + angle_offset)
        x = int(cx + radius * math.cos(rad))
        y = int(cy + radius * math.sin(rad))
        cv2.circle(frame, (x, y), 2, ORANGE, -1)

def draw_scan_circle(frame, cx, cy, radius):
    cv2.circle(frame, (cx, cy), radius, CYAN, 1)
    cv2.circle(frame, (cx, cy), radius + 10, ORANGE, 1)

# ================= LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Cinematic dark overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (5, 10, 25), -1)
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark

        cx = int((lm[0].x + lm[5].x + lm[17].x) / 3 * w)
        cy = int((lm[0].y + lm[5].y + lm[17].y) / 3 * h)

        fx = int(lm[8].x * w)
        fy = int(lm[8].y * h)

        draw_hologram_ring(frame, cx, cy, 70, angle)
        draw_hologram_ring(frame, cx, cy, 100, -angle)
        draw_scan_circle(frame, cx, cy, 130)

        if is_open_palm(lm) and not menu_active:
            menu_active = True
            login_opened = False
            particle = LoginParticle3D(cx, cy)

        if menu_active and particle:
            particle.update()

            hovered = particle.is_hovered(fx, fy)
            occluded = hovered and lm[8].z < -0.03

            # Color logic
            if hovered:
                if hover_start is None:
                    hover_start = time.time()
                elapsed = time.time() - hover_start
                color = RED if elapsed > 1 else ORANGE
            else:
                hover_start = None
                color = CYAN

            particle.draw(frame, color, occluded)

            if hovered:
                elapsed = time.time() - hover_start
                progress = int((elapsed / HOLD_TIME) * 360)

                cv2.ellipse(frame, (particle.x, particle.y),
                            (particle.r + 22, particle.r + 22),
                            0, 0, progress, RED, 3)

                if elapsed >= HOLD_TIME and not login_opened:
                    login_opened = True
                    path = os.path.abspath("login.html")
                    webbrowser.open(f"file:///{path}")

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        cv2.circle(frame, (fx, fy), 6, ORANGE, -1)

    cv2.putText(frame, "JARVIS // AUTHENTICATION",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, ORANGE, 2)

    angle += 2
    cv2.imshow("JARVIS IRON MAN UI", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
