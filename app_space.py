
import os
import time
import csv
import random
import math
import cv2
import mediapipe as mp
import numpy as np
from config_space import config

# === Global monotonic timestamp for MediaPipe VIDEO mode ===
_GLOBAL_T0 = time.time()
def _now_ms():
    return int((time.time() - _GLOBAL_T0) * 1000)

# ==== MediaPipe Hands (Tasks API) ====
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=config.model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

TIP = {"thumb":4, "index":8}
MCP = {"index":5, "pinky":17}

COLOR_BG    = (20,20,24)
COLOR_TEXT  = (255,255,255)
COLOR_ENEMY = (20, 50, 220)
COLOR_SHIP  = (40, 220, 220)
COLOR_BUL   = (230, 230, 230)
COLOR_HIT   = (0, 230, 0)

def l2(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    return math.hypot(dx, dy)

def pinch_distance(hand):
    return l2(hand[TIP["thumb"]], hand[TIP["index"]])

def hand_width(hand):
    return l2(hand[MCP["index"]], hand[MCP["pinky"]]) + 1e-6

def is_pinching(hand):
    return (pinch_distance(hand) / hand_width(hand)) < config.pinch_ratio_thr

class Enemy:
    __slots__ = ("x","y","alive")
    def __init__(self, x, y):
        self.x = x; self.y = y; self.alive = True

class Bullet:
    __slots__ = ("x","y","vy")
    def __init__(self, x, y, vy=-8):
        self.x = x; self.y = y; self.vy = vy

def draw_ship(frame, x, y):
    # triangle
    pts = np.array([[x, y-10],[x-12, y+10],[x+12, y+10]], np.int32)
    cv2.fillConvexPoly(frame, pts, COLOR_SHIP)

def draw_enemy(frame, x, y, alive=True):
    color = COLOR_ENEMY if alive else COLOR_HIT
    pts = np.array([[x, y-10],[x-10, y+10],[x+10, y+10]], np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

def AABB_hit(bx, by, ex, ey, radius=14):
    return (abs(bx-ex) <= radius) and (abs(by-ey) <= radius)

def save_csv(path, rows):
    header = ["timestamp","shots","hits","accuracy","enemies_destroyed","session_seconds"]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header: w.writerow(header)
        w.writerows(rows)

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    if config.fixed_width and config.fixed_height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.fixed_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.fixed_height)

    # Game world (screen coords in pixels)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 960
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 540

    # Enemies grid
    enemies = []
    cols, rows = config.enemy_cols, config.enemy_rows
    margin_x, margin_y = 70, 60
    gap_x = (W - 2*margin_x) // (cols-1) if cols>1 else 0
    gap_y = 40
    for j in range(rows):
        for i in range(cols):
            x = margin_x + i*gap_x
            y = margin_y + j*gap_y
            enemies.append(Enemy(x,y))

    # Ship
    ship_x = W//2; ship_y = int(H*0.85)
    ship_speed = 12

    # Bullets & shooting
    bullets = []
    last_shot = 0
    shots = 0
    hits = 0

    # Enemy horizontal movement
    dir_sign = 1
    last_step = time.time()

    start = time.time()
    end_time = start + config.session_seconds
    flash_until = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if config.mirror: frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        # overlay a faint bg
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (w,h), COLOR_BG, -1)
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

        # Hand control
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, _now_ms())

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            # map index tip x to ship_x
            ix = int(hand[TIP["index"]].x * w)
            ship_x = int(0.8*ship_x + 0.2*ix)   # smoothing
            # shoot with pinch
            if is_pinching(hand) and (time.time() - last_shot) > config.shoot_cooldown:
                bullets.append(Bullet(ship_x, ship_y-16, vy=-config.bullet_speed))
                last_shot = time.time()
                shots += 1
            # draw minimal hand landmarks for feedback
            for lm in [hand[TIP["index"]], hand[TIP["thumb"]]]:
                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 4, (0,255,0), -1)

        # Move enemies horizontally
        if time.time() - last_step > config.enemy_step_seconds:
            last_step = time.time()
            for e in enemies:
                if e.alive:
                    e.x += dir_sign * config.enemy_step_px
            # bounce at edges
            xs = [e.x for e in enemies if e.alive]
            if xs:
                if max(xs) > w - 30 or min(xs) < 30:
                    dir_sign *= -1
                    for e in enemies:
                        if e.alive: e.y += config.enemy_drop_px

        # Move bullets
        for b in bullets:
            b.y += b.vy
        bullets = [b for b in bullets if b.y > -10]

        # Collisions
        for b in bullets:
            for e in enemies:
                if e.alive and AABB_hit(b.x, b.y, e.x, e.y, radius=14):
                    e.alive = False
                    hits += 1
                    b.y = -9999  # mark for remove
                    flash_until = time.time() + 0.08
                    break
        bullets = [b for b in bullets if b.y > -100]

        # Draw enemies
        for e in enemies:
            draw_enemy(frame, e.x, e.y, e.alive)

        # Draw bullets
        for b in bullets:
            cv2.line(frame, (b.x, b.y), (b.x, b.y-10), COLOR_BUL, 2)

        # Draw ship
        draw_ship(frame, ship_x, ship_y)

        # HUD
        now = time.time()
        time_left = max(0.0, end_time - now)
        alive = sum(1 for e in enemies if e.alive)
        acc = (100.0*hits/shots) if shots>0 else 0.0

        cv2.rectangle(frame, (0,0), (w, 60), (0,0,0), -1)
        cv2.putText(frame, f"SPACE Time: {time_left:05.1f}s  Hits: {hits}  Shots: {shots}  Acc: {acc:04.1f}%  Left: {alive}",
                    (10,35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
        if time.time() < flash_until:
            cv2.putText(frame, "HIT!", (w-120, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_HIT, 2)

        cv2.imshow("SPACE Hand Landmarker", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break  # ESC
        time.sleep(config.frame_sleep)

        # end conditions
        if time_left <= 0 or alive == 0:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save metrics
    os.makedirs(config.metrics_dir, exist_ok=True)
    out_csv = os.path.join(config.metrics_dir, f"space_metrics.csv")
    rows = [[time.strftime("%Y-%m-%d %H:%M:%S"), shots, hits, (hits/shots if shots else 0.0), (len([1 for e in enemies if not e.alive])), config.session_seconds]]
    save_csv(out_csv, rows)

    print("=== SESSION SUMMARY ===")
    print(f"Shots: {shots}  Hits: {hits}  Acc: {(100.0*hits/shots if shots else 0.0):.1f}%")
    print(f"Enemies destroyed: {len([1 for e in enemies if not e.alive])}")
    print(f"Metrics saved to: {out_csv}")
