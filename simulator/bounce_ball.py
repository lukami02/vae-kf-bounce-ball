import numpy as np
import cv2
import imageio
import sys
sys.path.append("..")
from config.simulation_config import SimulationConfig

class BouncingBallSim:

    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.H, self.W = config.size
        self.gravity = config.gravity
        self.rng = np.random.default_rng(config.seed)

        # physics scale
        scale = min(self.H, self.W)

        self.r = max(1, int(scale * config.ball_scale))
        self.ball_sigma = scale * config.ball_sigma
        self.speed_range = (scale * config.speed_range[0], scale * config.speed_range[1])

        self.obstacle_min = int(scale * config.obstacle_min_scale)
        self.obstacle_max = int(scale * config.obstacle_max_scale)

        self.gravity_strength = scale * config.gravity_strength_scale

        self.obstacles = np.zeros((0,4), dtype=np.int32)

    def random_obstacles(self):
        """
        Obstacle generation
        """
        obs = []

        min_gap = 6*self.r

        n = self.rng.integers(self.cfg.num_obstacles+1)
        walls = ['left', 'right', 'top', 'bottom']
        last_wall = None

        opposite = {
            'left': 'right',
            'right': 'left',
            'top': 'bottom',
            'bottom': 'top'
        }

        for _ in range(n):

            if last_wall is None:
                wall = walls[self.rng.integers(len(walls))]
            else:
                wall = opposite[last_wall]

            depth = self.rng.integers(self.obstacle_min, self.obstacle_max)

            if wall in ('left', 'right'):
                h = self.rng.integers(self.obstacle_min, self.obstacle_max)
                y = self.rng.integers(self.r *2, self.H - self.r*2 - h)

                if wall == 'left':
                    x1, y1, x2, y2 = 0, y, depth, y+h
                else:
                    x1, y1, x2, y2 = self.W - depth, y, self.W, y+h
                
                if wall == 'left' and x2 > self.W - min_gap:
                    continue
                if wall == 'right' and x1 < min_gap:
                    continue

            else:
                w = self.rng.integers(self.obstacle_min, self.obstacle_max)
                x = self.rng.integers(self.r*2, self.W - self.r*2 - w)

                if wall == 'top':
                    x1, y1, x2, y2 = x, 0, x+w, depth
                else:
                    x1, y1, x2, y2 = x, self.H - depth, x+w, self.H

                if wall == 'top' and y2 > self.H - min_gap:
                    continue
                if wall == 'bottom' and y1 < min_gap:
                    continue
            obs.append((x1,y1,x2,y2))
            last_wall = wall

        self.obstacles = np.array(obs, dtype=np.int32)

    def step(self, pos, vel, apply_gravity=False):
        """
        Update position and velocity using substeps for more stable collision physics.
        """
        dt = 1.0 / self.cfg.substeps

        for _ in range(self.cfg.substeps):
            # Apply gravity to actual velocity
            if apply_gravity:
                vel[1] += self.gravity_strength * dt

            pos += vel * dt

            # Wall collisions
            if pos[0] < self.r:
                pos[0] = self.r
                vel[0] *= -1
            if pos[0] > self.W - self.r:
                pos[0] = self.W - self.r
                vel[0] *= -1

            if pos[1] < self.r:
                pos[1] = self.r
                vel[1] *= -1
            if pos[1] > self.H - self.r:
                pos[1] = self.H - self.r
                vel[1] *= -1

            # Obstacle collisions
            for x1,y1,x2,y2 in self.obstacles:
                cx = np.clip(pos[0], x1, x2)
                cy = np.clip(pos[1], y1, y2)

                dx = pos[0] - cx
                dy = pos[1] - cy
                dist_sq = dx*dx + dy*dy

                if dist_sq >= self.r*self.r:
                    continue
                
                overlap_x = (x2 - x1) / 2 + self.r - abs(pos[0] - (x1 + x2) / 2)
                overlap_y = (y2 - y1) / 2 + self.r - abs(pos[1] - (y1 + y2) / 2)

                if overlap_x <= 0 or overlap_y <= 0:
                    continue

                if overlap_x < overlap_y:
                    # Resolve along X — push ball horizontally
                    vel[0] *= -1
                    if pos[0] < (x1 + x2) / 2:
                        pos[0] = x1 - self.r
                    else:
                        pos[0] = x2 + self.r
                else:
                    # Resolve along Y — push ball vertically
                    vel[1] *= -1
                    if pos[1] < (y1 + y2) / 2:
                        pos[1] = y1 - self.r
                    else:
                        pos[1] = y2 + self.r
        return pos, vel

    def render_obstacles(self):
        """
        Renders frame with obstacles.
        """
        img = np.zeros((self.H, self.W), dtype=np.float32)

        for r in self.obstacles:
            cv2.rectangle(img, (r[0],r[1]), (r[2],r[3]), 1.0, -1, cv2.LINE_AA)

        return img

    def render_ball(self, pos):
        """
        Renders the current frame with the moving object.
        """
        x0,y0 = pos

        img = np.zeros((self.H,self.W), dtype=np.float32)

        if self.cfg.ball_gaussian:
            r = int(3*self.ball_sigma)

            x_min = max(0, int(x0-r))
            x_max = min(self.W, int(x0+r)+1)

            y_min = max(0, int(y0-r))
            y_max = min(self.H, int(y0+r)+1)

            x = np.arange(x_min,x_max)
            y = np.arange(y_min,y_max)

            xx,yy = np.meshgrid(x,y)

            g = np.exp(-((xx-x0)**2 + (yy-y0)**2) / (2*self.ball_sigma**2))
            g /= g.max()

            img[y_min:y_max,x_min:x_max] = g
        else:
            r = self.r + 0.2
            iy, ix = np.ogrid[0:self.H, 0:self.W]
            mask = (ix - int(x0)) ** 2 + (iy - int(y0)) ** 2 <= r * r
            img[mask] = 1.0
        return img
    
    def spawn_ball(self):
        """
        Randomly spawns the ball in the environment, avoiding collisions with obstacles.
        """
        for _ in range(20):

            pos = np.array([self.rng.uniform(self.r,self.W-self.r), self.rng.uniform(self.r,self.H-self.r)])
            good = True

            for r in self.obstacles:
                if ( r[0]-2*self.r < pos[0] < r[2]+2*self.r and r[1]-2*self.r < pos[1] < r[3]+2*self.r):
                    good = False
                    break

            if good:
                return pos

        return np.array([self.W/2,self.H/2])

    def generate_episode(self):
        """
        Generates a sequence of the simulation.
        """
        T = self.cfg.T
        apply_gravity = self.gravity and self.rng.random() < self.cfg.gravity_chance
        pos = self.spawn_ball()

        angle = self.rng.uniform(0, 2*np.pi)
        speed = self.rng.uniform(*self.speed_range)

        vel = np.array([ np.cos(angle)*speed, np.sin(angle)*speed])

        obstacle_frame = self.render_obstacles()
        balls = []

        for _ in range(T):
            ball = self.render_ball(pos)
            balls.append(ball)
            pos, vel = self.step(pos, vel, apply_gravity)

        balls = np.stack(balls)

        frames = np.concatenate([balls,], axis=0)

        control_val = 1.0 if apply_gravity else 0.0
        control_signal = np.full((T, 1), control_val, dtype=np.float32)

        return frames, obstacle_frame, control_signal

    def generate_dataset(self, seed=None):
        """
        Creates a large-scale dataset of physics simulations.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        ball_data = np.zeros((self.cfg.episodes, self.cfg.T, self.H, self.W), dtype=np.float32)
        obstacle_data = np.zeros((self.cfg.episodes, self.H, self.W), dtype=np.float32)
        control_data = np.zeros((self.cfg.episodes, self.cfg.T, 1), dtype=np.float32)

        for i in range(self.cfg.episodes):
            self.random_obstacles()
            frames, obstacle_frame, ctrl_signal = self.generate_episode()

            ball_data[i] = frames
            obstacle_data[i] = obstacle_frame 
            control_data[i] = ctrl_signal

            if i % 1000 == 0:
                print(f"Generated episode {i+1}/{self.cfg.episodes}")

        return ball_data, obstacle_data, control_data
    
if __name__ == "__main__":

    cfg = SimulationConfig()
    # Simulation parameters
    output_file = "ball.gif"

    # Initialize simulation
    sim = BouncingBallSim(cfg)
    sim.random_obstacles()

    # Generate animation
    ball_frames, obstacle = sim.generate_episode()
    ball_frames = np.maximum(ball_frames, obstacle[None, :, :])

    # Save GIF
    imageio.mimsave(output_file, (ball_frames * 255).astype(np.uint8), fps=10)

    print(f"Simulation complete. Saved to {output_file}")