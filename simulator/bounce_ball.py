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
        for _ in range(self.rng.integers(self.cfg.num_obstacles+1)):
            w = self.rng.integers(self.obstacle_min, self.obstacle_max)
            h = self.rng.integers(self.obstacle_min, self.obstacle_max)

            x = self.rng.integers(0, self.W - w)
            y = self.rng.integers(0, self.H - h)

            obs.append([x, y, x+w, y+h])

        self.obstacles = np.array(obs, dtype=np.int32)

    def step(self, pos, vel):
        """
        Update position and velocity using substeps for more stable collision physics.
        """
        dt = 1.0 / self.cfg.substeps

        for _ in range(self.cfg.substeps):
            # Apply gravity to actual velocity
            if self.gravity:
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

                if dist_sq < self.r*self.r:
                    dist = np.sqrt(dist_sq) + 1e-6
                    nx = dx / dist
                    ny = dy / dist
                    penetration = self.r - dist
                    pos[0] += nx * penetration
                    pos[1] += ny * penetration

                    vel -= 2 * np.dot(vel, [nx,ny]) * np.array([nx,ny])

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
            r = 3*self.r
            cv2.circle(img, center=(int(x0), int(y0)), radius=int(r), color=1.0, thickness=-1)
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
        pos = self.spawn_ball()

        angle = self.rng.uniform(0, 2*np.pi)
        speed = self.rng.uniform(*self.speed_range)

        vel = np.array([ np.cos(angle)*speed, np.sin(angle)*speed])

        obstacle_frame = self.render_obstacles()
        balls = []

        for _ in range(T):
            ball = self.render_ball(pos)
            balls.append(ball)
            pos, vel = self.step(pos, vel)

        balls = np.stack(balls)

        # motion channel
        #motion = np.zeros_like(balls)
        #motion[1:] = balls[1:] - balls[:-1]

        frames = np.concatenate([balls,], axis=0)

        return frames, obstacle_frame

    def generate_dataset(self):
        """
        Creates a large-scale dataset of physics simulations.
        """
        ball_data = np.zeros((self.cfg.episodes, self.cfg.T, self.H, self.W), dtype=np.float32)
        obstacle_data = np.zeros((self.cfg.episodes, self.H, self.W), dtype=np.float32)

        for i in range(self.cfg.episodes):
            self.random_obstacles()
            frames, obstacle_frame = self.generate_episode()

            ball_data[i] = frames
            obstacle_data[i] = obstacle_frame 

            if i % 1000 == 0:
                print(f"Generated episode {i+1}/{self.cfg.episodes}")

        return ball_data, obstacle_data
    
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