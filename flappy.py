"""Flappy Bird controlled by lateral raises via webcam pose detection."""

import cv2
import pygame
import random
import numpy as np
import os
from dataclasses import dataclass
from enum import Enum

WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 700
GAME_WIDTH = 400  # Classic Flappy Bird was narrower
CAM_WIDTH = 1000  # More room for camera

GRAVITY = 0.35
FLAP_STRENGTH = -7
PIPE_GAP = 180
PIPE_WIDTH = 52
PIPE_SPEED = 2.5
PIPE_SPAWN_INTERVAL = 3500  # ms
GROUND_HEIGHT = 112
RAISE_THRESHOLD = 0


class GameState(Enum):
    MENU = 1
    PLAYING = 2
    GAME_OVER = 3


class Assets:
    def __init__(self):
        assets_path = os.path.join(os.path.dirname(__file__), 'assets')
        self.bird_frames = [
            pygame.image.load(os.path.join(assets_path, 'yellowbird-downflap.png')).convert_alpha(),
            pygame.image.load(os.path.join(assets_path, 'yellowbird-midflap.png')).convert_alpha(),
            pygame.image.load(os.path.join(assets_path, 'yellowbird-upflap.png')).convert_alpha(),
        ]
        self.bird_frames = [pygame.transform.scale2x(f) for f in self.bird_frames]
        pipe_img = pygame.image.load(os.path.join(assets_path, 'pipe-green.png')).convert_alpha()
        self.pipe_bottom = pygame.transform.scale2x(pipe_img)
        self.pipe_top = pygame.transform.flip(self.pipe_bottom, False, True)
        bg_img = pygame.image.load(os.path.join(assets_path, 'background-day.png')).convert()
        self.background = pygame.transform.scale(bg_img, (GAME_WIDTH, WINDOW_HEIGHT))
        base_img = pygame.image.load(os.path.join(assets_path, 'base.png')).convert()
        self.ground = pygame.transform.scale(base_img, (GAME_WIDTH * 2, GROUND_HEIGHT))
        self.numbers = []
        for i in range(10):
            num_img = pygame.image.load(os.path.join(assets_path, f'{i}.png')).convert_alpha()
            self.numbers.append(pygame.transform.scale2x(num_img))
        self.gameover = pygame.image.load(os.path.join(assets_path, 'gameover.png')).convert_alpha()
        self.gameover = pygame.transform.scale2x(self.gameover)
        self.message = pygame.image.load(os.path.join(assets_path, 'message.png')).convert_alpha()
        self.message = pygame.transform.scale(self.message, 
            (int(self.message.get_width() * 1.5), int(self.message.get_height() * 1.5)))


@dataclass
class Bird:
    x: float = GAME_WIDTH // 3
    y: float = WINDOW_HEIGHT // 2 - 50
    velocity: float = 0
    frame_index: int = 0
    animation_timer: int = 0
    
    def flap(self):
        self.velocity = FLAP_STRENGTH
    
    def update(self):
        self.velocity += GRAVITY
        self.y += self.velocity
        self.animation_timer += 1
        if self.animation_timer >= 5:
            self.animation_timer = 0
            self.frame_index = (self.frame_index + 1) % 3
    
    def get_rect(self, assets):
        frame = assets.bird_frames[self.frame_index]
        return pygame.Rect(self.x - frame.get_width()//2 + 3,
                          self.y - frame.get_height()//2 + 3,
                          frame.get_width() - 6, frame.get_height() - 6)
    
    def draw(self, surface, assets):
        frame = assets.bird_frames[self.frame_index]
        rotation = max(-25, min(-self.velocity * 4, 70))
        rotated = pygame.transform.rotate(frame, rotation)
        rect = rotated.get_rect(center=(self.x, self.y))
        surface.blit(rotated, rect)


@dataclass
class Pipe:
    x: float
    gap_y: float
    passed: bool = False
    
    def update(self):
        self.x -= PIPE_SPEED
    
    def get_rects(self, assets):
        pipe_height = assets.pipe_bottom.get_height()
        top_bottom = self.gap_y - PIPE_GAP // 2
        top_rect = pygame.Rect(self.x, top_bottom - pipe_height, PIPE_WIDTH * 2, pipe_height)
        bottom_top = self.gap_y + PIPE_GAP // 2
        bottom_rect = pygame.Rect(self.x, bottom_top, PIPE_WIDTH * 2, pipe_height)
        
        return top_rect, bottom_rect
    
    def draw(self, surface, assets):
        pipe_height = assets.pipe_bottom.get_height()
        bottom_y = self.gap_y + PIPE_GAP // 2
        surface.blit(assets.pipe_bottom, (self.x, bottom_y))
        top_y = self.gap_y - PIPE_GAP // 2 - pipe_height
        surface.blit(assets.pipe_top, (self.x, top_y))


class PoseDetector:
    def __init__(self):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        import urllib.request
        
        model_path = "pose_landmarker_lite.task"
        if not os.path.exists(model_path):
            print("Downloading pose model...")
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
            urllib.request.urlretrieve(url, model_path)
            print("Model downloaded!")
        
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        
        self.left_raised = False
        self.right_raised = False
        self.raise_triggered = False
        self.was_raised = False
        self.frame_count = 0
    
    def detect(self, frame):
        import mediapipe as mp
        
        self.frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        self.left_raised = False
        self.right_raised = False
        raise_now = False
        
        timestamp_ms = int(self.frame_count * 1000 / 30)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            landmarks = result.pose_landmarks[0]
            
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            
            if left_wrist.y < left_shoulder.y - RAISE_THRESHOLD:
                self.left_raised = True
            if right_wrist.y < right_shoulder.y - RAISE_THRESHOLD:
                self.right_raised = True
            
            raise_now = self.left_raised or self.right_raised

            h, w = frame.shape[:2]
            connections = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
            
            for c in connections:
                if c[0] < len(landmarks) and c[1] < len(landmarks):
                    pt1 = (int(landmarks[c[0]].x * w), int(landmarks[c[0]].y * h))
                    pt2 = (int(landmarks[c[1]].x * w), int(landmarks[c[1]].y * h))
                    cv2.line(frame, pt1, pt2, (255, 255, 255), 2)

            arm_indices = [11, 12, 13, 14, 15, 16]
            for idx in arm_indices:
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    pt = (int(lm.x * w), int(lm.y * h))
                    cv2.circle(frame, pt, 8, (0, 255, 0), -1)
        
        self.raise_triggered = raise_now and not self.was_raised
        self.was_raised = raise_now
        
        return frame
    
    def should_flap(self):
        return self.raise_triggered


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Flappy Bird - Lateral Raise Control")
        
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.assets = Assets()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.pose_detector = PoseDetector()
        self.state = GameState.MENU
        self.bird = Bird()
        self.pipes = []
        self.score = 0
        self.last_pipe_time = 0
        self.ground_scroll = 0
    
    def reset(self):
        self.bird = Bird()
        self.pipes = []
        self.score = 0
        self.last_pipe_time = pygame.time.get_ticks()
        self.state = GameState.PLAYING
    
    def spawn_pipe(self):
        gap_y = random.randint(150, WINDOW_HEIGHT - GROUND_HEIGHT - 150)
        self.pipes.append(Pipe(x=GAME_WIDTH + 50, gap_y=gap_y))
    
    def check_collision(self):
        bird_rect = self.bird.get_rect(self.assets)
        if self.bird.y < 0 or self.bird.y > WINDOW_HEIGHT - GROUND_HEIGHT - 20:
            return True
        for pipe in self.pipes:
            top_rect, bottom_rect = pipe.get_rects(self.assets)
            if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect):
                return True
        
        return False
    
    def update(self):
        if self.state != GameState.PLAYING:
            self.bird.animation_timer += 1
            if self.bird.animation_timer >= 5:
                self.bird.animation_timer = 0
                self.bird.frame_index = (self.bird.frame_index + 1) % 3
            return
        
        self.bird.update()
        
        current_time = pygame.time.get_ticks()
        if current_time - self.last_pipe_time > PIPE_SPAWN_INTERVAL:
            self.spawn_pipe()
            self.last_pipe_time = current_time
        
        for pipe in self.pipes:
            pipe.update()
            if not pipe.passed and pipe.x + PIPE_WIDTH * 2 < self.bird.x:
                pipe.passed = True
                self.score += 1
        
        self.pipes = [p for p in self.pipes if p.x > -PIPE_WIDTH * 2]
        self.ground_scroll = (self.ground_scroll + PIPE_SPEED) % 24
        
        if self.check_collision():
            self.state = GameState.GAME_OVER

    def draw_score(self, surface):
        score_str = str(self.score)
        total_width = sum(self.assets.numbers[int(d)].get_width() for d in score_str)
        x = (GAME_WIDTH - total_width) // 2
        
        for digit in score_str:
            num_img = self.assets.numbers[int(digit)]
            surface.blit(num_img, (x, 50))
            x += num_img.get_width()
    
    def draw_game(self):
        game_surface = pygame.Surface((GAME_WIDTH, WINDOW_HEIGHT))
        game_surface.blit(self.assets.background, (0, 0))

        for pipe in self.pipes:
            pipe.draw(game_surface, self.assets)
        game_surface.blit(self.assets.ground, (-self.ground_scroll, WINDOW_HEIGHT - GROUND_HEIGHT))
        self.bird.draw(game_surface, self.assets)
        if self.state == GameState.PLAYING:
            self.draw_score(game_surface)
        if self.state == GameState.MENU:
            msg_rect = self.assets.message.get_rect(center=(GAME_WIDTH // 2, WINDOW_HEIGHT // 2 - 50))
            game_surface.blit(self.assets.message, msg_rect)
        elif self.state == GameState.GAME_OVER:
            go_rect = self.assets.gameover.get_rect(center=(GAME_WIDTH // 2, WINDOW_HEIGHT // 3))
            game_surface.blit(self.assets.gameover, go_rect)
            self.draw_score(game_surface)
        
        self.screen.blit(game_surface, (0, 0))
    
    def draw_camera(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (CAM_WIDTH, WINDOW_HEIGHT))
        frame = cv2.flip(frame, 1)
        frame_surface = pygame.surfarray.make_surface(np.rot90(frame))
        self.screen.blit(frame_surface, (GAME_WIDTH, 0))

        left_color = (100, 255, 100) if self.pose_detector.left_raised else (80, 80, 80)
        right_color = (100, 255, 100) if self.pose_detector.right_raised else (80, 80, 80)
        
        pygame.draw.rect(self.screen, left_color, (GAME_WIDTH + 50, 30, 60, 80), border_radius=10)
        pygame.draw.rect(self.screen, (255,255,255), (GAME_WIDTH + 50, 30, 60, 80), 3, border_radius=10)
        pygame.draw.rect(self.screen, right_color, (GAME_WIDTH + CAM_WIDTH - 110, 30, 60, 80), border_radius=10)
        pygame.draw.rect(self.screen, (255,255,255), (GAME_WIDTH + CAM_WIDTH - 110, 30, 60, 80), 3, border_radius=10)
        
        left_text = self.font.render("L", True, (255, 255, 255))
        right_text = self.font.render("R", True, (255, 255, 255))
        self.screen.blit(left_text, (GAME_WIDTH + 70, 55))
        self.screen.blit(right_text, (GAME_WIDTH + CAM_WIDTH - 90, 55))
        status = "Raise your arms!"
        if self.pose_detector.left_raised and self.pose_detector.right_raised:
            status = "BOTH ARMS UP!"
        elif self.pose_detector.left_raised:
            status = "LEFT ARM UP!"
        elif self.pose_detector.right_raised:
            status = "RIGHT ARM UP!"
        
        status_text = self.font.render(status, True, (0, 255, 0))
        self.screen.blit(status_text, 
                        (GAME_WIDTH + CAM_WIDTH // 2 - status_text.get_width() // 2, 
                         WINDOW_HEIGHT - 50))
    
    def run(self):
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        if self.state == GameState.PLAYING:
                            self.bird.flap()
                        else:
                            self.reset()
                    elif event.key == pygame.K_ESCAPE:
                        running = False
            
            ret, frame = self.cap.read()
            if ret:
                frame = self.pose_detector.detect(frame)
                if self.pose_detector.should_flap():
                    if self.state == GameState.PLAYING:
                        self.bird.flap()
                    else:
                        self.reset()
            
            self.update()
            self.draw_game()
            
            if ret:
                self.draw_camera(frame)
            else:
                pygame.draw.rect(self.screen, (30, 30, 50), 
                               (GAME_WIDTH, 0, CAM_WIDTH, WINDOW_HEIGHT))
                no_cam = self.font.render("Camera not found", True, (255, 100, 100))
                self.screen.blit(no_cam, (GAME_WIDTH + CAM_WIDTH // 2 - no_cam.get_width() // 2,
                                         WINDOW_HEIGHT // 2))
            
            pygame.display.flip()
            self.clock.tick(60)
        
        self.cap.release()
        pygame.quit()


if __name__ == "__main__":
    game = Game()
    game.run()
