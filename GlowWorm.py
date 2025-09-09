### Distributed Glowworm Optimization Scheme
### Wiki: https://en.wikipedia.org/wiki/Glowworm_swarm_optimization
import numpy as np
import matplotlib.pyplot as plt
import simpy
import random
import multiprocessing

##############################################################################
# GSO PARAMETERS 
##############################################################################
dims = 1000
nturns = 6000
max_jitter = 0.2
initialDeploymentList = ['grid','random','spiral']


##############################################################################
# Fitness Function 
##############################################################################
def fitness_function(x, y): # Three-Hump Camel
    x1 = x/180 - 2.9
    x2 = y/180 - 2.9
    # The formula for the Three-Hump Camel function
    f_x = np.log(2 * x1**2 - 1.05 * x1**4 + (x1**6) / 6 + x1 * x2 + x2**2)
    return (5-f_x)/10



##############################################################################
# Min Max of the fitness function
##############################################################################
x, y = np.array(np.meshgrid(np.linspace(0, dims, dims), np.linspace(0, dims, dims)))
z = fitness_function(x, y)
x_min, y_min = x.ravel()[z.argmin()], y.ravel()[z.argmin()]
x_max, y_max = x.ravel()[z.argmax()], y.ravel()[z.argmax()]
min_fitness = fitness_function(x_min, y_min)
max_fitness = fitness_function(x_max, y_max)



##############################################################################
# Keep the swarm do not go beyond the boundary
##############################################################################
def keep_in_bounds(x, dims):
    if x < 0:
        return 0
    elif x > dims-10:
        return dims-10
    else:
        return x
    


"""
Compute the polar coordinates of (x1, y1) relative to (x2, y2).

Parameters
----------
(x1, y1) : numpy tuple of float
    Coordinates of the target point.
(x2, y2) : numpy tuple of float
    Coordinates of the reference point.

Returns
-------
(r, theta) : numpy tuple of floats
    r is the distance from (x1, y1) to (x2, y2).
    theta is the angle (in radians) from the positive x-axis.
"""

def to_relative_polar(x1, x2):
    dx = x1[0] - x2[0]
    dy = x1[1] - x2[1]
    
    r = np.sqrt(dx**2 + dy**2)      # Radial distance
    theta = np.arctan2(dy, dx)       # Angle in radians

    polar = np.array([r, theta])
    
    return polar

"""
Compute the cartesian coordinate of (x1, y1) when polar coordinate and cartesian coordinate of (x2, y2) are given.

Parameters
----------
(r, theta) : numpy tuple of floats
    r is the distance from (x1, y1) to (x2, y2).
    theta is the angle (in radians) from the positive x-axis.
(x2, y2) : numpy tuple of float
    Coordinates of the reference point.

Returns
-------
(x1, y1) : numpy tuple of float
    Coordinates of the target point.
"""

def to_cartesian(polar, x2):
    dx = x2[0] + np.cos(polar[1]) * polar[0]
    dy = x2[1] + np.sin(polar[1]) * polar[0]

    x1 = np.array([dx, dy])
    
    return x1


def weighted_random_choice(population, weights):
    """
    Selects a random element from a list with associated probabilities.

    Args:
        population: A sequence (list, tuple, etc.) of elements to choose from.
        weights: A sequence of weights, with each weight corresponding to
            the probability of selecting the element at the same index
            in the population. Weights must be non-negative and can be
            integers or floating-point numbers.

    Returns:
        A randomly selected element from the population, based on the weights.
    """
    #print(len(population), len(weights))
    if len(population) != len(weights):
        raise ValueError("Population and weights must have the same length")
    if any(w < 0 for w in weights):
      raise ValueError("Weights cannot be negative")

    return random.choices(population, weights=weights)[0]


##############################################################################
# GSO FUNCTIONS 
##############################################################################
class Glowworm:
    def __init__(self, env, name, swarm, X=np.array([1, 1]), waypoint = np.array([1, 1]),  speed=1.5433, 
                 transRange=190, score = 0.0, errorRate = 0.0):
        self.env = env
        self.name = name
        self.swarm = swarm  # List of all particles
        self.transmission_range = transRange
        self.speed = speed # 0.015433 -> 3 knots
        self.waypoint = waypoint
        self.errorRate = errorRate

        self.X = X
        self.V = np.random.randn(2) * 0.1
        self.score = score # for transmisstion distance
        self.influenceTable = {name: {'distance':0.0, 'angle':0.0, 'score':0.0}} # name: distance, angle, score

        self.positions = [self.X.copy()]
        self.process = env.process(self.run())


    def sensing(self): # 
        """accquire the fitness of worm's current position to calculate 'influence radius'."""
        self.score = fitness_function(self.X[0],self.X[1])
        # if found a better score
        if self.score > self.influenceTable[self.name]['score']:
            self.influenceTable[self.name]['score'] = self.score
            self.influenceTable[self.name]['angle'] = 0.0
            self.influenceTable[self.name]['distance'] = 0.0


    def prepareMsgToSend(self):
        """Prepare to send a portion of the influence table to neighboring AUVs"""
        msgInfluenceTable = {}
        for key, value in self.influenceTable.items():
            if value['score'] >= self.score: # only send the entries with higher score than self
                msgInfluenceTable[key] = value
        return msgInfluenceTable


    def broadcast(self):
        """If worm j is within worm j's radius, record distance; else 0. For influence martrix"""
        for glowworm in self.swarm:
            distance =  np.linalg.norm(self.X-glowworm.X)
            if distance <= self.transmission_range: # Changed from self.score
                if glowworm.name != self.name:
                    polar_cor = to_relative_polar(self.X, glowworm.X)
                    msgInfluenceTable = self.prepareMsgToSend()
                    if np.random.rand() >= self.errorRate: # failure to recieve if not
                        glowworm.receive(self.name, msgInfluenceTable, polar_cor) 


    def receive(self, worm_name, worm_msgInfluenceTable, polar_cor):
        """update the influenceTable"""
        for key, value in worm_msgInfluenceTable.items():
            tableEntry_name = key
            tableEntry_distance = value['distance']
            tableEntry_score = value['score']
            tableEntry_angle = value['angle']
            tableEntry_polar = np.array([tableEntry_distance,tableEntry_angle])
            tableEntrytoSelf = to_relative_polar(to_cartesian(tableEntry_polar,np.array([0,0])) + to_cartesian(polar_cor,np.array([0,0])), np.array([0,0]))
            if (tableEntry_name not in self.influenceTable) or (self.influenceTable[tableEntry_name]['score'] < tableEntry_score):
                self.influenceTable[tableEntry_name] = {'distance':tableEntrytoSelf[0], 'angle':tableEntrytoSelf[1], 'score':tableEntry_score}
            
    def nextWaypoint(self): 
        """Calculate the worm's next waypoint based on the Table of influences."""
        """Choose the best worm"""
        new_position = np.array([0.0, 0.0])
        potentialWormList = []
        scoreDiffList = []
        for glowworm_name in self.influenceTable:
            glowworm_score = self.influenceTable[glowworm_name]['score']
            glowworm_angle = self.influenceTable[glowworm_name]['angle']
            glowworm_distance = self.influenceTable[glowworm_name]['distance']
            if glowworm_distance != 0 and self.score < glowworm_score:
                scoreDiffList.append(glowworm_score - self.score)
                potentialWormList.append([glowworm_distance,glowworm_angle])

        if len(scoreDiffList) != 0:
            # Make choice only the best one
            chosenWorm = potentialWormList[scoreDiffList.index(max(scoreDiffList))]
            new_position = to_cartesian(chosenWorm, self.X)
            
            # Bound checking
            new_position[0] = keep_in_bounds(new_position[0], dims)
            new_position[1] = keep_in_bounds(new_position[1], dims)

            self.waypoint = new_position


    def move(self):  ### working on
        distance2Waypoint = np.linalg.norm(self.X - self.waypoint) # Euclidean distance
        if distance2Waypoint <= 3:  # Within in the threshold of waypoint
            self.nextWaypoint()     # Calculate next waypoint but do not move
        else: # make a move when still far from waypoint
            direction = self.waypoint-self.X
            newPosition = self.X + self.speed * (direction / np.linalg.norm(direction))
            for key, value in self.influenceTable.items(): # update influence table's polar cordinates
                prevPolar = np.array([value['distance'],value['angle']])
                nerghborX = to_cartesian(prevPolar,self.X)
                newPolar = to_relative_polar(nerghborX,newPosition)
                self.influenceTable[key]['distance'] = newPolar[0]
                self.influenceTable[key]['angle'] = newPolar[1]
            self.X = newPosition


    ##############################################################################
    # SIMPY INTEGRATION
    ##############################################################################
    def run(self):
        """
        A generator (process) that runs GSO for 'nturns' iterations in a SimPy environment.
        Each iteration is considered one time-step in the simulation.
        """
        i = 0
        while True:
            #print("Time ", i)
            # Compute glowworm logic
            self.sensing()
            if i%5 == 0:
                self.broadcast()
            self.move()

            # Store positions for later analysis or plotting
            if i % 10 == 0:    
                self.positions.append(self.X.copy()) # take the records of position
                #print("name: ", self.name, " - score: ", self.score)
            i = i + 1

            # Advance simulated time by 1 time unit
            yield self.env.timeout(1)


def starting_points_random(num_worms):
    """Initialize the worm positions randomly."""
    return np.random.rand(num_worms, 2) * dims

def starting_points_grid(num_worms):
    """Evenly distribute particles on a grid within the search space."""
    list_glowworm = []
    start = 0.1 * dims
    end = 0.9 * dims
    
    # Determine the number of rows and columns based on the number of particles
    n_side = int(np.ceil(np.sqrt(num_worms)))  # Number of rows and columns in the grid
    x_coords = np.linspace(start, end, n_side)
    y_coords = np.linspace(start, end, n_side)

    # Create particles and assign grid positions
    for i in range(num_worms):
        # Calculate grid coordinates (i.e., row and column)
        row = i // n_side
        col = i % n_side
        grid_position = np.array([x_coords[col], y_coords[row]])  # Assign grid position
        list_glowworm.append(grid_position)
    
    return np.array(list_glowworm)

def starting_points_spiral(num_nodes: int):
    """
    Calculates and plots the positions of n nodes on an Archimedean spiral.

    The spiral is centered in a square area and scaled to fit within its
    boundaries.

    Args:
        num_nodes (int): The total number of nodes (AUVs) to plot.
    """
    # --- 1. Define Spiral Parameters ---
    # The spiral is centered at (0,0), so the max radius is half the side length.
    max_radius = dims * 0.45
    
    # We define the spiral's "tightness" by the number of full rotations.
    # More rotations create a denser spiral.
    num_rotations = 10 
    
    # Determine the spiral constant 'a' which controls the distance between arms.
    # The formula is R = a * theta. We want R=max_radius at theta_max.
    theta_max = num_rotations * 2 * np.pi
    a = max_radius / theta_max
    
    # --- 2. Calculate Node Positions ---
    # Create an array of angles, one for each node, from 0 to theta_max.
    theta = np.linspace(0, theta_max, num_nodes)
    
    # Calculate the radius for each node based on its angle.
    R = a * theta
    
    # Convert from polar coordinates (R, theta) to Cartesian (x, y).
    x_coords = R * np.cos(theta) + dims/2
    y_coords = R * np.sin(theta) + dims/2

    return np.stack((x_coords, y_coords), axis=1)

def check_termination_condition(sim_env, particles, target_position, threshold=1.0, radius=5):
    while True:
        count_within_radius = sum(
            np.linalg.norm(p.X - target_position) <= radius for p in particles
        )
        if (count_within_radius >= threshold * len(particles)) or (sim_env.now >= nturns):
            #print(f"Condition met at time {env.now}: {count_within_radius}/{len(particles)} particles near target.")
            return [sim_env.event().succeed(),count_within_radius/len(particles)]  # Trigger the event to stop simulation
        yield sim_env.timeout(1)


def run_gso_simpy(AUVnum=25,transmissionRange=190,initialDeployment=0,LinkerrorRate = 0.0):
    """
    Sets up the SimPy environment, runs the glowworm process,
    and returns the recorded positions.
    """
    sim_env = simpy.Environment()

    # Initial population
    num_worms = AUVnum
    T_R = transmissionRange
    if initialDeployment == 0:
        pop = starting_points_grid(num_worms)
    elif initialDeployment == 1:
        pop = starting_points_random(num_worms)
    elif initialDeployment == 2:
        pop = starting_points_spiral(num_worms)
    else:
        pop = []

    # Create and start the GSO process
    swarm = []    
    for i in range(num_worms):
        swarm.append(Glowworm(env = sim_env, name = i, transRange = transmissionRange, swarm = [], 
                              X=pop[i], waypoint = pop[i]+np.random.uniform(-50, 50, size=2),errorRate = LinkerrorRate))

    for glowworm in swarm:
        glowworm.swarm = swarm  # Give each glowworm the list of all glowworm
        #particle.target = [target_position,target_fitness]

    # Run the simulation
    termination_event = sim_env.process(check_termination_condition(sim_env, swarm, np.array([x_max, y_max])))
    sim_env.run(until=termination_event)  # Simulation stops when termination_event is triggered

    swarm_positions = []
    for glowworm in swarm:
        swarm_positions.append(glowworm.positions)

    count_within_radius = sum(np.linalg.norm(glowworm.X - np.array([x_max, y_max])) <= 10 for glowworm in swarm)

    return swarm_positions,count_within_radius/num_worms, sim_env.now


def worker_function(AUVnum=25,transmissionRange=300,initialDeployment=0,LinkerrorRate = 0.0):
    # Perform CPU-bound computation on 'data'
    AggregatePercentageList = []
    AggregateDurationList = []
    roundOfSimulation = 10
    for i in range(roundOfSimulation):
        all_positions,AUVpercentage,sim_duration = run_gso_simpy(AUVnum,transmissionRange,initialDeployment,LinkerrorRate)
        AggregatePercentageList.append(AUVpercentage)
        AggregateDurationList.append(sim_duration)
        print(AUVpercentage, sim_duration)
    avgAggregatePercentage = np.mean(AggregatePercentageList)
    avgAggregateDuration = np.mean(AggregateDurationList)
    print('Initial Deployment: ', initialDeploymentList[initialDeployment], ' TR: ',transmissionRange,
          ' AUV_NUM: ',AUVnum, ' AVG %: ', avgAggregatePercentage, ' AVG Duration: ',avgAggregateDuration)
    return avgAggregatePercentage,avgAggregateDuration


##############################################################################
# MAIN (Demo)
##############################################################################
if __name__ == "__main__":
    processes = []
    results = []
    #AUVnum = 25
    #numProcesses = 20
    transmissionRange = 300
    LinkerrorRateList = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    #initialDeploymentList = [0,1,2]
    initialDeployment = 0
    
    print("Simulation Cap: ", nturns)

    for LinkerrorRate in LinkerrorRateList:
        for AUVnum in [25,36,49,64,81,100]:
            p = multiprocessing.Process(target=worker_function, args=(AUVnum,transmissionRange,initialDeployment,LinkerrorRate,))
            processes.append(p)
            p.start()

    for p in processes:
        p.join() # Wait for processes to complete
        # In a real scenario, you'd use queues or pipes to get results back