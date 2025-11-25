
import os

class SpaceConfig:
    def __init__(self):
        # Modelo de manos
        self.model_path = os.path.join(os.path.dirname(__file__), "models/hand_landmarker.task")

        # Cámara/UI
        self.mirror = True
        self.fixed_width = 960
        self.fixed_height = 540
        self.frame_sleep = 0.01

        # Control por mano
        self.pinch_ratio_thr = 0.45   # (dist pulgar-indice) / (ancho mano) para disparar
        self.shoot_cooldown = 0.22    # s entre disparos

        # Gameplay
        self.session_seconds = 60.0
        self.bullet_speed = 14
        self.enemy_cols = 6
        self.enemy_rows = 3
        self.enemy_step_seconds = 0.25
        self.enemy_step_px = 12
        self.enemy_drop_px = 10

        # Métricas
        self.metrics_dir = os.path.join(os.path.dirname(__file__), "metrics")

config = SpaceConfig()
