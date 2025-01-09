import simpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def f(x, y):
    "Objective function"
    return ((x/100)-3.14)**2 + ((y/100)-2.72)**2 + np.sin(3*(x/100)+1.41) + np.sin(4*(y/100)-1.73)

class Particle:
    def __init__(self, env, name, particles, X=np.array([1, 1]), target = [np.array([250,250]),999], waypoint = np.array([250,250]), c1=0.1, c2=0.1, w=0.8, speed=1.5433, transRange=100.0):
        self.env = env
        self.name = name
        self.particles = particles  # List of all particles
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.transmission_range = transRange
        self.speed = speed # 0.015433 -> 3 knots
        self.stop_flag = False # when find the target stop
        self.targetFounded = False
        self.waypoint = waypoint

        self.X = X
        self.target = target
        self.V = np.random.randn(2) * 0.1
        self.pbest_position = self.X.copy()
        self.pbest_fitness = f(self.X[0], self.X[1])

        self.nbest_position = self.X.copy()
        self.nbest_fitness = self.pbest_fitness

        self.positions = [self.X.copy()]

        self.process = env.process(self.run())

    def broadcast(self):
        """Broadcast nbest to other particles based on proximity."""
        for particle in self.particles:
            if particle != self:
                distance = np.linalg.norm(self.X - particle.X) # Euclidean distance
                if distance < self.transmission_range:  # Accoustic module Transmission range
                    particle.receive(self.nbest_position, self.nbest_fitness, self.targetFounded)
                    # if self.targetFounded:
                    #     particle.receive(self.target[0][0], self.target[0][1], self.targetFounded)
                    # else:
                    #     particle.receive(self.nbest_position, self.nbest_fitness, self.targetFounded)

    def receive(self, nbest_position, nbest_fitness, isTarget):
        """update the personal nbest"""
        if nbest_fitness < self.nbest_fitness:
            self.nbest_position = nbest_position.copy()
            self.nbest_fitness = nbest_fitness
            self.targetFounded = isTarget
    
    def update(self):
        r1, r2 = np.random.rand(2)
        self.V = self.w * self.V + self.c1 * r1 * (self.pbest_position - self.X) + self.c2 * r2 * (self.nbest_position - self.X)
        self.waypoint = self.X + self.V

    def sensing(self):
        if not self.targetFounded:
            fitness = f(self.X[0], self.X[1])

            if fitness < self.pbest_fitness:
                self.pbest_position = self.X.copy()
                self.pbest_fitness = fitness

            if fitness < self.nbest_fitness:
                self.nbest_position = self.X.copy()
                self.nbest_fitness = fitness

    def move(self):
        if not self.stop_flag:
            if self.targetFounded:
                direction = self.nbest_position-self.X
            else:
                direction = self.waypoint-self.X
            self.X = self.X + self.speed * (direction / np.linalg.norm(direction))
            distance = np.linalg.norm(self.X - self.waypoint) # Euclidean distance
            if distance <= 3:  # Interaction threshold
                self.update()
                #print(self.name," Updated! ", self.V, " nbest_loc ", self.nbest_position, " nbest_fit ", self.nbest_fitness)

    def targetFound(self):
        if np.linalg.norm(self.X - self.target[0]) <= 1:
            self.stop_flag = True
            self.targetFounded = True
            self.nbest_position = self.target[0].copy()
            self.nbest_fitness = self.target[1]

    def run(self):
        i = 0
        while True:
            self.targetFound()
            self.sensing()
            self.move()
            if i%5 ==0:
                self.broadcast()  # broadcast nbest to neighbors
            if i%10 ==0:    
                self.positions.append(self.X.copy()) # take the records of position
            i = i + 1
            yield self.env.timeout(1)

def initialize_arc_particles(n_particles, radius):
    """Evenly distribute particles on an arc from 0 to pi/2."""
    particles_postions = []
    
    start_angle = (np.pi / 2)*(2/90)
    end_angle = (np.pi / 2)*(88/90)

    # Generate evenly spaced angles between 0 and pi/2
    angles = np.linspace(start_angle, end_angle, n_particles)

    # Create particles and assign positions based on the arc
    for i, theta in enumerate(angles):
        # Convert polar coordinates to Cartesian (x, y)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        position = np.array([x, y])  # Particle position on the arc
        particles_postions.append(position)
    
    return particles_postions

def check_termination_condition(env, particles, target_position, threshold=1.0, radius=5):
    while True:
        count_within_radius = sum(
            np.linalg.norm(p.X - target_position) <= radius for p in particles
        )
        if count_within_radius >= threshold * len(particles):
            print(f"Condition met at time {env.now}: {count_within_radius}/{len(particles)} particles near target.")
            return env.event().succeed()  # Trigger the event to stop simulation
        yield env.timeout(1)

def simulate_pso():
    env = simpy.Environment()
    n_particles = 20
    radius = 500
    firstWaypoints = initialize_arc_particles(n_particles, radius)
    
    x, y = np.array(np.meshgrid(np.linspace(0, 500, 5000), np.linspace(0, 500, 5000)))
    z = f(x, y)
    x_min, y_min = x.ravel()[z.argmin()], y.ravel()[z.argmin()]
    target_position = np.array([x_min, y_min])  # Define the target position
    target_fitness = f(x_min, y_min)

    particles = [Particle(env, f'Particle {i}', [], waypoint=firstWaypoints[i]) for i in range(n_particles)]
    for particle in particles:
        particle.particles = particles  # Give each particle the list of all particles
        particle.target = [target_position,target_fitness]

    termination_event = env.process(check_termination_condition(env, particles, target_position))
    env.run(until=termination_event)  # Simulation stops when termination_event is triggered

    global_best = min(particles, key=lambda p: p.pbest_fitness)
    return [p.positions for p in particles], global_best.pbest_position, global_best.pbest_fitness