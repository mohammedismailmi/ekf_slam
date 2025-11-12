import pygame
import numpy as np
import math
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import threading
import json
from flask import render_template, send_from_directory, abort
import os

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Pygame and simulation parameters
WIDTH, HEIGHT = 1200, 800
FPS = 30
GRID_SIZE = 20  # For occupancy grid

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 100, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
CYAN = (0, 255, 255)
PURPLE = (200, 0, 255)

class Camera:
    """Simulates a monocular camera with field of view"""
    def __init__(self, fov=90, max_range=300):
        self.fov = math.radians(fov)  # Field of view in radians
        self.max_range = max_range
        self.num_rays = 15  # Number of sampling rays
        
    def get_visible_landmarks(self, robot_pos, robot_angle, landmarks, walls):
        """Returns landmarks visible within camera FOV, checking for occlusion."""
        visible = []
        rx, ry = robot_pos
        
        for lm_id, (lx, ly) in landmarks.items():
            # Calculate relative position
            dx = lx - rx
            dy = ly - ry
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance > self.max_range:
                continue

            # --- 1. Check for Occlusion (Line of Sight) ---
            is_occluded = False
            for wall in walls:
                # clipline returns a tuple of points if the line intersects the rect
                if wall.clipline((rx, ry), (lx, ly)):
                    is_occluded = True
                    break
            
            if is_occluded:
                continue
            
            # --- 2. Check Field of View ---
            # Calculate bearing
            bearing = math.atan2(dy, dx)
            angle_diff = self.normalize_angle(bearing - robot_angle)
            
            # Check if within FOV
            if abs(angle_diff) <= self.fov / 2:
                # --- GAUSSIAN ERROR (MEASUREMENT) ---
                # This was already here: noise is added to sensor readings
                noise_dist = np.random.normal(0, 2)     # std dev of 2 pixels
                noise_bearing = np.random.normal(0, 0.02) # std dev of ~1.15 degrees
                # ------------------------------------
                
                visible.append({
                    'id': lm_id,
                    'distance': distance + noise_dist,
                    'bearing': angle_diff + noise_bearing,
                    'position': (lx, ly)
                })
        
        return visible
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

class EKF_SLAM:
    """Extended Kalman Filter for SLAM"""
    def __init__(self, initial_pos):
        # State: [x, y, theta, landmark1_x, landmark1_y, ...]
        self.state = np.array([initial_pos[0], initial_pos[1], 0.0])
        
        # Covariance matrix
        self.P = np.eye(3) * 0.1
        
        # Process noise (model of the noise we added to Robot.update)
        self.Q = np.diag([0.1, 0.1, 0.05])
        
        # Measurement noise (model of the noise in Camera.get_visible_landmarks)
        self.R = np.diag([4.0, 0.04])  # distance, bearing
        
        # Landmark dictionary
        self.landmarks = {}
        self.landmark_count = 0
        
    def predict(self, v, w, dt):
        """Prediction step with motion model"""
        x, y, theta = self.state[0], self.state[1], self.state[2]
        
        # Motion model (using commanded v, w)
        if abs(w) < 0.001:
            # Straight line motion
            x_new = x + v * math.cos(theta) * dt
            y_new = y + v * math.sin(theta) * dt
            theta_new = theta
        else:
            # Curved motion
            x_new = x + (v/w) * (math.sin(theta + w*dt) - math.sin(theta))
            y_new = y + (v/w) * (-math.cos(theta + w*dt) + math.cos(theta))
            theta_new = theta + w * dt
        
        theta_new = self.normalize_angle(theta_new)
        
        # Update state
        self.state[0] = x_new
        self.state[1] = y_new
        self.state[2] = theta_new
        
        # Jacobian of motion model
        G = np.eye(len(self.state))
        G[0, 2] = -v * math.sin(theta) * dt
        G[1, 2] = v * math.cos(theta) * dt
        
        # Update covariance
        Q_full = np.zeros((len(self.state), len(self.state)))
        Q_full[0:3, 0:3] = self.Q
        
        self.P = G @ self.P @ G.T + Q_full
    
    def update(self, observations):
        """Update step with landmark observations"""
        for obs in observations:
            lm_id = obs['id']
            z_dist = obs['distance']
            z_bearing = obs['bearing']
            
            if lm_id not in self.landmarks:
                # Initialize new landmark
                self.initialize_landmark(lm_id, z_dist, z_bearing)
            else:
                # Update existing landmark
                self.update_landmark(lm_id, z_dist, z_bearing)
    
    def initialize_landmark(self, lm_id, distance, bearing):
        """Add new landmark to state"""
        x, y, theta = self.state[0], self.state[1], self.state[2]
        
        # Calculate landmark position
        lm_x = x + distance * math.cos(theta + bearing)
        lm_y = y + distance * math.sin(theta + bearing)
        
        # Add to state
        self.state = np.append(self.state, [lm_x, lm_y])
        
        # Expand covariance matrix
        n = len(self.state)
        P_new = np.zeros((n, n))
        P_new[:-2, :-2] = self.P
        P_new[-2, -2] = 100  # Initial uncertainty
        P_new[-1, -1] = 100
        self.P = P_new
        
        # Store landmark index
        self.landmarks[lm_id] = len(self.state) - 2
        
    def update_landmark(self, lm_id, z_dist, z_bearing):
        """Update landmark estimate using EKF"""
        idx = self.landmarks[lm_id]
        
        x, y, theta = self.state[0], self.state[1], self.state[2]
        lm_x, lm_y = self.state[idx], self.state[idx+1]
        
        # Expected measurement
        dx = lm_x - x
        dy = lm_y - y
        q = dx**2 + dy**2
        z_dist_exp = math.sqrt(q)
        z_bearing_exp = self.normalize_angle(math.atan2(dy, dx) - theta)
        
        # Innovation
        z = np.array([z_dist, z_bearing])
        z_exp = np.array([z_dist_exp, z_bearing_exp])
        y_inn = z - z_exp
        y_inn[1] = self.normalize_angle(y_inn[1])
        
        # Jacobian
        H = np.zeros((2, len(self.state)))
        sqrt_q = math.sqrt(q)
        
        H[0, 0] = -dx / sqrt_q
        H[0, 1] = -dy / sqrt_q
        H[0, idx] = dx / sqrt_q
        H[0, idx+1] = dy / sqrt_q
        
        H[1, 0] = dy / q
        H[1, 1] = -dx / q
        H[1, 2] = -1
        H[1, idx] = -dy / q
        H[1, idx+1] = dx / q
        
        # Kalman gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state and covariance
        self.state = self.state + K @ y_inn
        self.state[2] = self.normalize_angle(self.state[2])
        self.P = (np.eye(len(self.state)) - K @ H) @ self.P
    
    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def get_estimated_landmarks(self):
        """Return estimated landmark positions"""
        est_landmarks = {}
        for lm_id, idx in self.landmarks.items():
            est_landmarks[lm_id] = (self.state[idx], self.state[idx+1])
        return est_landmarks

class OccupancyGrid:
    """Grid-based occupancy mapping"""
    def __init__(self, width, height, cell_size, offset_x, offset_y):
        self.width = width // cell_size
        self.height = height // cell_size
        self.cell_size = cell_size
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.grid = np.ones((self.height, self.width)) * 0.5  # Unknown = 0.5
        self.current_raw_hits = [] # To store real-time wall hits (x, y)
        self.current_grid_hits = set() # To store (gx, gy) of hits for mini-map
        
    def update(self, robot_pos, robot_angle, camera, walls):
        """Update occupancy grid using camera rays"""
        rx, ry = robot_pos
        self.current_raw_hits = [] # Clear hits from last frame
        self.current_grid_hits.clear() # Clear grid hits from last frame
        
        for i in range(camera.num_rays):
            # Calculate ray angle
            angle_offset = (i / (camera.num_rays - 1) - 0.5) * camera.fov
            ray_angle = robot_angle + angle_offset
            
            # Ray casting
            for dist in range(0, int(camera.max_range), 5):
                x = rx + dist * math.cos(ray_angle)
                y = ry + dist * math.sin(ray_angle)
                
                # Convert to grid coordinates, accounting for offset
                gx = int((x - self.offset_x) / self.cell_size)
                gy = int((y - self.offset_y) / self.cell_size)
                
                if 0 <= gx < self.width and 0 <= gy < self.height:
                    # Check if hit wall
                    hit_wall = False
                    for wall in walls:
                        if wall.collidepoint(x, y):
                            hit_wall = True
                            # Increase certainty of occupied cell
                            self.grid[gy, gx] = min(1.0, self.grid[gy, gx] + 0.1)
                            self.current_raw_hits.append((x, y)) # Store the raw hit point
                            self.current_grid_hits.add((gx, gy)) # Store the grid cell hit
                            break
                    
                    if hit_wall:
                        break # Stop this ray
                    else:
                        # Free space - decrease certainty of occupied
                        self.grid[gy, gx] = max(0.0, self.grid[gy, gx] - 0.05)
    
    def draw(self, screen):
        """Draw occupancy grid"""
        for y in range(self.height):
            for x in range(self.width):
                prob = self.grid[y, x]
                # 0 (free) = black, 1 (occupied) = white
                color_val = int(prob * 255)
                color = (color_val, color_val, color_val)
                
                rect = pygame.Rect(
                    self.offset_x + x * self.cell_size,
                    self.offset_y + y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(screen, color, rect)

class MiniMap:
    """A persistent mini-map that traces discovered walls."""
    def __init__(self, x, y, width, height, maze_width, maze_height, grid_size):
        self.x = x  # Top-left corner on screen
        self.y = y
        self.width = width
        self.height = height
        self.grid_size = grid_size
        
        # Calculate scaling factor to fit maze (700x600) into map (350x300)
        self.scale_x = self.width / maze_width
        self.scale_y = self.height / maze_height
        self.scale = min(self.scale_x, self.scale_y) # Use smallest scale to fit
        
        # Calculate scaled grid cell size
        self.cell_draw_size = max(1, int(self.grid_size * self.scale))

        self.discovered_grid_points = set() # Stores (gx, gy) tuples

    def update(self, grid_hits):
        """Add new grid coordinates to the discovered set."""
        # grid_hits is a set of (gx, gy) tuples
        self.discovered_grid_points.update(grid_hits)

    def draw(self, screen):
        """Draw the mini-map overlay."""
        # Draw background and border
        map_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(screen, DARK_GRAY, map_rect)
        pygame.draw.rect(screen, WHITE, map_rect, 2)
        
        # Draw discovered wall cells
        for (gx, gy) in self.discovered_grid_points:
            # Calculate position within the maze (0,0) origin
            maze_x = gx * self.grid_size
            maze_y = gy * self.grid_size
            
            # Scale down and offset to mini-map position
            map_x = int(maze_x * self.scale)
            map_y = int(maze_y * self.scale)
            
            # Draw the cell
            draw_rect = pygame.Rect(
                self.x + map_x,
                self.y + map_y,
                self.cell_draw_size,
                self.cell_draw_size
            )
            # Clip drawing to be inside the map rectangle
            clipped_rect = draw_rect.clip(map_rect)
            if clipped_rect.width > 0 and clipped_rect.height > 0:
                pygame.draw.rect(screen, WHITE, clipped_rect)

class Robot:
    """Robot with camera"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0
        self.v = 0  # Linear velocity
        self.w = 0  # Angular velocity
        self.size = 15 # This is the RADIUS
        self.trail = []
        self.max_trail = 200
        
        # --- NEW ---
        # Define standard deviations for motion noise
        # Feel free to tune these values
        self.v_noise_std = 2.0   # Noise in linear velocity (units/sec)
        self.w_noise_std = 0.05  # Noise in angular velocity (rad/sec)
        # -----------
        
    def update(self, dt, walls):
        """Update robot position with collision detection"""
        
        # --- UPDATED: ADD GAUSSIAN ERROR (MOTION) ---
        # Add noise to the commanded velocities to simulate imperfect motors
        v_noisy = self.v + np.random.normal(0, self.v_noise_std)
        w_noisy = self.w + np.random.normal(0, self.w_noise_std)
        # ---------------------------------------------
        
        # Calculate potential new state
        x, y, theta = self.x, self.y, self.angle
        v, w = v_noisy, w_noisy # Use the noisy values for the update
        
        if abs(w) < 0.001:
            x_new = x + v * math.cos(theta) * dt
            y_new = y + v * math.sin(theta) * dt
            theta_new = theta
        else:
            x_new = x + (v/w) * (math.sin(theta + w*dt) - math.sin(theta))
            y_new = y + (v/w) * (-math.cos(theta + w*dt) + math.cos(theta))
            theta_new = theta + w * dt
        
        # Update angle always (robot can turn in place)
        self.angle = self.normalize_angle(theta_new)

        # --- Collision Detection ---
        # Create a bounding box for the new position.
        new_rect = pygame.Rect(x_new - self.size, y_new - self.size, self.size * 2, self.size * 2)
        
        # Check for collision
        if new_rect.collidelist(walls) == -1:
            # No collision, update position
            self.x = x_new
            self.y = y_new
        else:
            # Collision! Stop linear motion.
            self.v = 0
            # We don't update x and y, so the robot stays in its last valid position.

        # Add to trail (using the *actual* final position)
        self.trail.append((self.x, self.y))
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)

    def normalize_angle(self, angle):
        """Normalize angle to [0, 2*pi] for consistency, though EKF uses [-pi, pi]"""
        angle = angle % (2 * math.pi)
        return angle
    
    def draw(self, screen):
        """Draw robot"""
        # Draw trail
        if len(self.trail) > 1:
            pygame.draw.lines(screen, BLUE, False, self.trail, 2)
        
        # Draw robot body
        pygame.draw.circle(screen, RED, (int(self.x), int(self.y)), self.size)
        
        # Draw direction indicator
        end_x = self.x + self.size * 1.5 * math.cos(self.angle)
        end_y = self.y + self.size * 1.5 * math.sin(self.angle)
        pygame.draw.line(screen, YELLOW, (self.x, self.y), (end_x, end_y), 3)

class SLAMSimulation:
    """Main SLAM simulation"""
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Camera-Based EKF-SLAM")
        self.clock = pygame.time.Clock()
        self.running = True
        self.paused = False
        
        # --- Create environment (maze) ---
        self.walls = [
            # Outer boundary
            pygame.Rect(50, 50, 700, 10),      # Top wall
            pygame.Rect(50, 50, 10, 600),      # Left wall
            pygame.Rect(50, 640, 700, 10),     # Bottom wall
            pygame.Rect(740, 50, 10, 600),     # Right wall

            # --- Maze Walls ---
            # First vertical barrier (forces robot right)
            pygame.Rect(200, 50, 10, 250),
            # Horizontal barrier blocking top path
            pygame.Rect(200, 300, 300, 10),
            # Vertical barrier from bottom (creates U-turn)
            pygame.Rect(350, 400, 10, 240),
            # Horizontal barrier from left
            pygame.Rect(50, 400, 200, 10),
            # Vertical barrier in middle
            pygame.Rect(500, 150, 10, 300),
            # Horizontal barrier from right (top)
            pygame.Rect(500, 150, 240, 10),
            # Horizontal barrier from right (lower)
            pygame.Rect(600, 550, 150, 10),
            # Vertical barrier (lower right)
            pygame.Rect(600, 400, 10, 150),
        ]
        
        # True landmarks (features in the environment)
        self.true_landmarks = {
            0: (150, 150), # Start area
            1: (150, 350), # By left wall, before turn
            2: (300, 250), # between first walls
            3: (300, 500), # lower-mid area
            4: (450, 100), # top-mid area
            5: (550, 300), # mid-right
            6: (700, 200), # top-right corner
            7: (650, 600), # end area
        }
        
        # Initialize robot (starts in top-left "room")
        self.robot = Robot(100, 100)
        
        # Initialize camera
        self.camera = Camera(fov=60, max_range=200)
        
        # Initialize EKF-SLAM
        self.ekf = EKF_SLAM((self.robot.x, self.robot.y))
        
        # Initialize occupancy grid
        self.maze_width = 700 # 750 - 50
        self.maze_height = 600 # 650 - 50
        self.maze_offset_x = 50
        self.maze_offset_y = 50
        self.occupancy_grid = OccupancyGrid(
            self.maze_width, 
            self.maze_height, 
            GRID_SIZE, 
            self.maze_offset_x, 
            self.maze_offset_y
        )
        
        # Initialize Mini-Map
        self.mini_map = MiniMap(
            x=800, y=220, width=350, height=300, # Position and size on screen
            maze_width=self.maze_width,          # Actual maze dimensions
            maze_height=self.maze_height,
            grid_size=GRID_SIZE
        )

        # UI state
        self.show_fov = True
        self.show_grid = True
        self.show_features = True
        
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                # --- NEW: Keyboard controls ---
                elif event.key == pygame.K_UP:
                    self.robot.v = 50
                elif event.key == pygame.K_DOWN:
                    self.robot.v = -50
                elif event.key == pygame.K_LEFT:
                    self.robot.w = 1.0
                elif event.key == pygame.K_RIGHT:
                    self.robot.w = -1.0
            # --- NEW: Handle key releases ---
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                    self.robot.v = 0
                elif event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                    self.robot.w = 0
    
    def update(self):
        """Update simulation"""
        if self.paused:
            return
        
        dt = 1.0 / FPS
        
        # Update robot (now with collision detection and motion noise)
        self.robot.update(dt, self.walls)
        
        # EKF Prediction
        # The EKF is given the *commanded* velocity, not the noisy one.
        # This is correct: the filter knows your command, and its
        # process noise (Q) models the *uncertainty* in that command.
        self.ekf.predict(self.robot.v, self.robot.w, dt)
        
        # Get camera observations (now with occlusion)
        observations = self.camera.get_visible_landmarks(
            (self.robot.x, self.robot.y),
            self.robot.angle,
            self.true_landmarks,
            self.walls
        )
        
        # EKF Update
        self.ekf.update(observations)
        
        # Update occupancy grid
        # Run update if grid OR fov is shown (fov needs the hit data)
        if self.show_grid or self.show_fov:
            self.occupancy_grid.update(
                (self.robot.x, self.robot.y),
                self.robot.angle,
                self.camera,
                self.walls
            )
            # Update the mini-map with the new hits
            self.mini_map.update(self.occupancy_grid.current_grid_hits)
    
    def draw(self):
        """Draw everything"""
        self.screen.fill(DARK_GRAY) # User-provided background
        
        # Draw occupancy grid (if enabled)
        if self.show_grid:
            self.occupancy_grid.draw(self.screen)
        
        # Draw walls
        for wall in self.walls:
            pygame.draw.rect(self.screen, WHITE, wall)
        
        # Draw camera FOV
        if self.show_fov:
            self.draw_camera_fov()
        
        # Draw true landmarks
            if self.show_features:
                for lm_id, (x, y) in self.true_landmarks.items():
                    pygame.draw.circle(self.screen, GREEN, (int(x), int(y)), 6)
                    font = pygame.font.Font(None, 20)
                    text = font.render(str(lm_id), True, WHITE) # Text color
                    self.screen.blit(text, (x + 10, y - 10))
        
        # Draw camera wall-hit highlights
        if self.show_fov:
            for (x, y) in self.occupancy_grid.current_raw_hits:
                pygame.draw.circle(self.screen, RED, (int(x), int(y)), 3)

        # Draw estimated landmarks
        estimated_landmarks = self.ekf.get_estimated_landmarks()
        for lm_id, (x, y) in estimated_landmarks.items():
            pygame.draw.circle(self.screen, PURPLE, (int(x), int(y)), 6, 2)
            
            # Draw uncertainty ellipse
            if lm_id in self.ekf.landmarks:
                idx = self.ekf.landmarks[lm_id]
                cov = self.ekf.P[idx:idx+2, idx:idx+2]
                self.draw_uncertainty_ellipse(x, y, cov)
        
        # Draw robot
        self.robot.draw(self.screen)
        
        # Draw EKF estimated position
        ekf_x, ekf_y = self.ekf.state[0], self.ekf.state[1]
        pygame.draw.circle(self.screen, CYAN, (int(ekf_x), int(ekf_y)), 8, 2)
        
        # Draw info panel
        self.draw_info_panel()
        
        # Draw Mini-Map
        self.mini_map.draw(self.screen)
        
        # Draw controls
        self.draw_controls()
        
        pygame.display.flip()
    
    def draw_camera_fov(self):
        """Draw camera field of view"""
        rx, ry = self.robot.x, self.robot.y
        angle = self.robot.angle
        fov = self.camera.fov
        max_range = self.camera.max_range
        
        # Draw FOV cone
        points = [(rx, ry)]
        for i in range(20):
            a = angle - fov/2 + (i / 19) * fov
            x = rx + max_range * math.cos(a)
            y = ry + max_range * math.sin(a)
            points.append((x, y))
        points.append((rx, ry))
        
        # Create surface with transparency
        fov_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(fov_surface, (255, 255, 0, 30), points)
        self.screen.blit(fov_surface, (0, 0))
        
        # Draw FOV boundary
        pygame.draw.lines(fov_surface, YELLOW, False, points, 2)
        self.screen.blit(fov_surface, (0, 0))
    
    def draw_uncertainty_ellipse(self, x, y, cov):
        """Draw uncertainty ellipse for landmark"""
        try:
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            angle = math.atan2(eigenvectors[1, 0], eigenvectors[0, 0])
            width = 2 * math.sqrt(eigenvalues[0]) * 3  # 3-sigma
            height = 2 * math.sqrt(eigenvalues[1]) * 3
            
            # Draw ellipse (simplified as circle for pygame)
            radius = int(max(width, height) / 2)
            if radius > 0 and radius < 500:  # Sanity check
                pygame.draw.circle(self.screen, PURPLE, (int(x), int(y)), radius, 1)
        except:
            pass
    
    def draw_info_panel(self):
        """Draw information panel"""
        font = pygame.font.Font(None, 24)
        y_offset = 50
        
        info = [
            f"Robot Pos: ({self.robot.x:.1f}, {self.robot.y:.1f})",
            f"Robot Angle: {math.degrees(self.robot.angle):.1f}°",
            f"EKF Pos: ({self.ekf.state[0]:.1f}, {self.ekf.state[1]:.1f})",
            f"Landmarks: {len(self.ekf.landmarks)}",
            f"Status: {'PAUSED' if self.paused else 'RUNNING'}",
        ]
        
        for i, text in enumerate(info):
            surface = font.render(text, True, WHITE) # CHANGED (from user's BLACK)
            self.screen.blit(surface, (800, y_offset + i * 30))
    
    def draw_controls(self):
        """Draw control instructions"""
        font = pygame.font.Font(None, 20)
        y_offset = 530 # Moved down to make space for mini-map
        
        controls = [
            "Controls:",
            "Arrow Keys - Move Robot", # NEW
            "SPACE - Pause/Resume",
            "Or use Flask web interface", # UPDATED
            "",
            "Features:",
            "1. Feature-based SLAM",
            "2. Occupancy Grid",
            "3. Camera FOV",
            "4. Uncertainty Ellipses",
        ]
        
        for i, text in enumerate(controls):
            surface = font.render(text, True, WHITE) # CHANGED (from user's BLACK)
            self.screen.blit(surface, (800, y_offset + i * 25))
    
    def run(self):
        """Main simulation loop"""
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        
        pygame.quit()

# Global simulation instance
sim = None

@app.route('/')
def index():
    # This file is not generated, but the Flask app expects it.
    # The user would need to create this 'index.html' separately.
    tpl_path = os.path.join(app.root_path, 'templates', 'index.html')
    if os.path.exists(tpl_path):
        return render_template('index.html')
    else:
        # Helpful fallback if file missing
        return "<h1>SLAM Control Interface</h1><p>templates/index.html not found. Create it inside the templates/ folder.</p>", 200

@app.route('/control', methods=['POST'])
def control():
    global sim
    if sim is None:
        return jsonify({'error': 'Simulation not running'}), 400
    
    data = request.json
    command = data.get('command')
    
    if command == 'forward':
        sim.robot.v = 50
    elif command == 'backward':
        sim.robot.v = -50
    elif command == 'left':
        sim.robot.w = 1.0
    elif command == 'right':
        sim.robot.w = -1.0
    elif command == 'stop':
        sim.robot.v = 0
        sim.robot.w = 0
    elif command == 'toggle_fov':
        sim.show_fov = not sim.show_fov
    elif command == 'toggle_grid':
        sim.show_grid = not sim.show_grid
    elif command == 'toggle_features':
        sim.show_features = not sim.show_features
    elif command == 'pause':
        sim.paused = not sim.paused
    
    return jsonify({'status': 'success'})

@app.route('/status')
def status():
    global sim
    if sim is None:
        return jsonify({'error': 'Simulation not running'}), 400
    
    return jsonify({
        'robot_pos': [float(sim.robot.x), float(sim.robot.y)],
        'robot_angle': float(sim.robot.angle),
        'ekf_pos': [float(sim.ekf.state[0]), float(sim.ekf.state[1])],
        'landmarks': len(sim.ekf.landmarks),
        'paused': sim.paused
    })

def run_flask():
    app.run(debug=False, port=5000, use_reloader=False)

def run_simulation():
    global sim
    sim = SLAMSimulation()
    sim.run()

if __name__ == '__main__':
    # Start Flask in separate thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    print("="*60)
    print("Camera-Based EKF-SLAM Simulation")
    print("="*60)
    print("Flask server running at: http://localhost:5000")
    print("Open the web interface to control the robot")
    print("="*60)
    
    # Run pygame simulation in main thread
    run_simulation()