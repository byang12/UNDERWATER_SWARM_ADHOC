##############################################################################
# GSO FUNCTIONS Only my own luciferin to neighbors
##############################################################################
class FireFly:
    def __init__(self, env, name, swarm, X=np.array([1, 1]), waypoint = np.array([1, 1]),  speed=1.5433, 
                 transRange=300, locationErrorRate=0.0, score = 0.0):
        self.env = env
        self.name = name
        self.swarm = swarm  # List of all particles
        self.transmission_range = transRange
        self.speed = speed # 0.015433 -> 3 knots
        self.waypoint = waypoint
        self.locationErrorRate = locationErrorRate
        

        self.X = X
        self.V = np.random.randn(2) * 0.1
        self.score = score # for transmisstion distance
        self.influenceTable = {name: {'distance':0.0, 'angle':0.0, 'score':0.0}} # name: distance, angle, score

        self.positions = [self.X.copy()]
        self.process = env.process(self.run())


    def sensing(self): # 
        """accquire the fitness of node's current position to calculate 'influence radius'."""
        self.score = fitness_function(self.X[0],self.X[1])
        # if found a better score
        if self.score > self.influenceTable[self.name]['score']:
            self.influenceTable[self.name]['score'] = self.score
            self.influenceTable[self.name]['angle'] = 0.0
            self.influenceTable[self.name]['distance'] = 0.0


    def prepareMsgToSend(self):
        """Prepare to send a portion of the influence table to neighboring AUVs"""
        # msgInfluenceTable = {}
        # for key, value in self.influenceTable.items():
        #     #print(value)
        #     if value['score'] >= self.score: # only send the entries with higher score than self
        #         msgInfluenceTable[key] = value

        msgInfluenceTable = {self.name: {'distance':0.0, 'angle':0.0, 'score':self.score}}

        #print(self.name)
        #print(msgInfluenceTable)
        return msgInfluenceTable


    def broadcast(self):
        """If node j is within node j's radius, record distance; else 0. For influence martrix"""
        for firefly in self.swarm:
            distance =  np.linalg.norm(self.X-firefly.X)
            if distance <= self.transmission_range: # Changed from self.score
                if firefly.name != self.name:
                    errorPoint = locating_error(self.X,firefly.X, self.locationErrorRate)
                    polar_cor = to_relative_polar(self.X, errorPoint)
                    msgInfluenceTable = self.prepareMsgToSend()
                    firefly.receive(self.name, msgInfluenceTable, polar_cor) 


    def receive(self, node_name, node_msgInfluenceTable, polar_cor):
        """update the influenceTable"""
        for key, value in node_msgInfluenceTable.items():
            tableEntry_name = key
            tableEntry_distance = value['distance']
            tableEntry_score = value['score']
            tableEntry_angle = value['angle']
            tableEntry_polar = np.array([tableEntry_distance,tableEntry_angle])
            tableEntrytoSelf = to_relative_polar(to_cartesian(tableEntry_polar,np.array([0,0])) + to_cartesian(polar_cor,np.array([0,0])), np.array([0,0]))
            if (tableEntry_name not in self.influenceTable) or (self.influenceTable[tableEntry_name]['score'] < tableEntry_score):
                self.influenceTable[tableEntry_name] = {'distance':tableEntrytoSelf[0], 'angle':tableEntrytoSelf[1], 'score':tableEntry_score}
            

    # def nextWaypoint(self):
    #     """Calculate the node's next waypoint based on the Table of influences."""
    #     """Weighted Randomly select the node from a group"""
    #     new_position = np.array([0.0, 0.0])
    #     potentialnodeList = []
    #     scoreDiffList = []
    #     for firefly_name in self.influenceTable:
    #         firefly_score = self.influenceTable[firefly_name]['score']
    #         firefly_angle = self.influenceTable[firefly_name]['angle']
    #         firefly_distance = self.influenceTable[firefly_name]['distance']
    #         if firefly_distance != 0 and self.score < firefly_score:
    #             scoreDiffList.append(firefly_score - self.score)
    #             potentialnodeList.append([firefly_distance,firefly_angle])

    #     if len(scoreDiffList) != 0:
    #         # Make a weighted random choice of node
    #         total = sum(scoreDiffList)
    #         probabilities = [element / total for element in scoreDiffList]
    #         chosennode = weighted_random_choice(potentialnodeList, probabilities)
    #         new_position = to_cartesian(chosennode, self.X)

    #         # Add random jitter
    #         jitter_x = max_jitter * np.random.rand() * np.random.randint(-100,200)
    #         jitter_y = max_jitter * np.random.rand() * np.random.randint(-100,200)
            
            
    #         new_position[0] = new_position[0] + jitter_x
    #         new_position[1] = new_position[1] + jitter_y
            
    #         # Bound checking
    #         new_position[0] = keep_in_bounds(new_position[0], dims)
    #         new_position[1] = keep_in_bounds(new_position[1], dims)

    #         self.waypoint = new_position


    def nextWaypoint(self): 
        """Calculate the node's next waypoint based on the Table of influences."""
        """Choose the best node"""
        new_position = np.array([0.0, 0.0])
        potentialnodeList = []
        scoreDiffList = []
        for firefly_name in self.influenceTable:
            firefly_score = self.influenceTable[firefly_name]['score']
            firefly_angle = self.influenceTable[firefly_name]['angle']
            firefly_distance = self.influenceTable[firefly_name]['distance']
            if firefly_distance != 0 and self.score < firefly_score:
                scoreDiffList.append(firefly_score - self.score)
                potentialnodeList.append([firefly_distance,firefly_angle])

        if len(scoreDiffList) != 0:
            # # Make a weighted random choice of node
            # total = sum(scoreDiffList)
            # probabilities = [element / total for element in scoreDiffList]
            # chosennode = weighted_random_choice(potentialnodeList, probabilities)
            # new_position = to_cartesian(chosennode, self.X)

            # Make choice only the best one
            chosennode = potentialnodeList[scoreDiffList.index(max(scoreDiffList))]
            new_position = to_cartesian(chosennode, self.X)
            
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
            # Compute firefly logic
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



########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################



##############################################################################
# FireFly Class & FUNCTIONS: Send all luciferin I know - Long running time
##############################################################################
class FireFly:
    def __init__(self, env, name, swarm, X=np.array([1, 1]), waypoint = np.array([1, 1]), broadcast_cycle=bc, 
                 speed=1.5433, transRange=150, locationErrorRate=0.0, score = 0.0):
        self.env = env
        self.name = name
        self.swarm = swarm  # List of all particles
        self.transmission_range = transRange
        self.speed = speed # 0.015433 -> 3 knots
        self.waypoint = waypoint
        self.locationErrorRate = locationErrorRate
        self.broadcast_cycle = broadcast_cycle
        

        self.X = X
        self.V = np.random.randn(2) * 0.1
        self.score = score # for transmisstion distance
        self.influenceTable = {name: {'distance':0.0, 'angle':0.0, 'score':0.0}} # name: distance, angle, score

        self.positions = [self.X.copy()]
        self.process = env.process(self.run())


    def sensing(self): # 
        """accquire the fitness of node's current position to calculate 'influence radius'."""
        self.score = fitness_function(self.X[0],self.X[1])
        # if found a better score
        if self.score > self.influenceTable[self.name]['score']:
            self.influenceTable[self.name]['score'] = self.score
            self.influenceTable[self.name]['angle'] = 0.0
            self.influenceTable[self.name]['distance'] = 0.0


    # def prepareMsgToSend(self):
    #     """Prepare to send a portion of the influence table to neighboring AUVs"""
    #     msgInfluenceTable = {}

    #     msgInfluenceTable = self.influenceTable # send all table

    #     # for key, value in self.influenceTable.items():
    #     #     if value['score'] >= self.score: # only send the entries with higher score than self
    #     #         msgInfluenceTable[key] = value

    #     # msgInfluenceTable = {self.name: {'distance':0.0, 'angle':0.0, 'score':self.score}}   # 

    #     #print(self.name)
    #     #print(msgInfluenceTable)
    #     return msgInfluenceTable


    def broadcast(self):
        """If node j is within node j's radius, record distance; else 0. For influence martrix"""
        for firefly in self.swarm:
            distance =  np.linalg.norm(self.X-firefly.X)
            if distance <= self.transmission_range: # Changed from self.score
                if firefly.name != self.name:
                    errorPoint = locating_error(self.X,firefly.X, self.locationErrorRate)
                    polar_cor = to_relative_polar(self.X, errorPoint)
                    # msgInfluenceTable = self.prepareMsgToSend()
                    msgInfluenceTable = self.influenceTable
                    firefly.receive(self.name, msgInfluenceTable, polar_cor) 


    def receive(self, node_name, node_msgInfluenceTable, polar_cor):
        """update the influenceTable"""
        for key, value in node_msgInfluenceTable.items():
            tableEntry_name = key
            tableEntry_distance = value['distance']
            tableEntry_score = value['score']
            tableEntry_angle = value['angle']
            tableEntry_polar = np.array([tableEntry_distance,tableEntry_angle])
            tableEntrytoSelf = to_relative_polar(to_cartesian(tableEntry_polar,np.array([0,0])) + to_cartesian(polar_cor,np.array([0,0])), np.array([0,0]))
            if (tableEntry_name not in self.influenceTable) or (self.influenceTable[tableEntry_name]['score'] < tableEntry_score):
                self.influenceTable[tableEntry_name] = {'distance':tableEntrytoSelf[0], 'angle':tableEntrytoSelf[1], 'score':tableEntry_score}
            

    def nextWaypoint(self): 
        """Calculate the node's next waypoint based on the Table of influences."""
        """Choose the best node"""
        new_position = np.array([0.0, 0.0])
        potentialnodeList = []
        scoreDiffList = []
        for firefly_name in self.influenceTable:
            firefly_score = self.influenceTable[firefly_name]['score']
            firefly_angle = self.influenceTable[firefly_name]['angle']
            firefly_distance = self.influenceTable[firefly_name]['distance']
            weaken_factor = int(firefly_distance/self.transmission_range)
            firefly_score = firefly_score*(gamma**weaken_factor) 
            if firefly_distance != 0 and self.score < firefly_score:
                scoreDiffList.append(firefly_score - self.score)
                potentialnodeList.append([firefly_distance,firefly_angle])

        if len(scoreDiffList) != 0:
            # # Make a weighted random choice of node
            # total = sum(scoreDiffList)
            # probabilities = [element / total for element in scoreDiffList]
            # chosennode = weighted_random_choice(potentialnodeList, probabilities)
            # new_position = to_cartesian(chosennode, self.X)

            # Make choice only the best one
            chosennode = potentialnodeList[scoreDiffList.index(max(scoreDiffList))]
            new_position = to_cartesian(chosennode, self.X)
            
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
            # Compute firefly logic
            self.sensing()
            if i%self.broadcast_cycle == 0:
                self.broadcast()
            self.move()

            # Store positions for later analysis or plotting
            if i % 10 == 0:    
                self.positions.append(self.X.copy()) # take the records of position
                #print("name: ", self.name, " - score: ", self.score)
            i = i + 1

            # Advance simulated time by 1 time unit
            yield self.env.timeout(1)



########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################




##############################################################################
# FireFly Class & FUNCTIONS -Work Take lesstime
##############################################################################
class FireFly:
    def __init__(self, env, name, swarm, X=np.array([1, 1]), waypoint = np.array([1, 1]), broadcast_cycle=bc, 
                 speed=1.5433, transRange=150, locationErrorRate=0.0, score = 0.0):
        self.env = env
        self.name = name
        self.swarm = swarm  # List of all particles
        self.transmission_range = transRange
        self.speed = speed # 0.015433 -> 3 knots
        self.waypoint = waypoint
        self.locationErrorRate = locationErrorRate
        self.broadcast_cycle = broadcast_cycle
        

        self.X = X
        self.V = np.random.randn(2) * 0.1
        self.score = score # for transmisstion distance
        self.influenceTable = {name: {'distance':0.0, 'angle':0.0, 'score':0.0}} # name: distance, angle, score

        self.positions = [self.X.copy()]
        self.process = env.process(self.run())


    def sensing(self): # 
        """accquire the fitness of node's current position to calculate 'influence radius'."""
        self.score = fitness_function(self.X[0],self.X[1])
        # if found a better score
        if self.score > self.influenceTable[self.name]['score']:
            self.influenceTable[self.name]['score'] = self.score
            self.influenceTable[self.name]['angle'] = 0.0
            self.influenceTable[self.name]['distance'] = 0.0


    def prepareMsgToSend(self):
        """Prepare to send a portion of the influence table to neighboring AUVs"""
        msgInfluenceTable = {}

        # msgInfluenceTable = self.influenceTable # send all table

        for key, value in self.influenceTable.items():
            if value['score'] >= self.score: # only send the entries with higher score than self
                msgInfluenceTable[key] = value

        # msgInfluenceTable = {self.name: {'distance':0.0, 'angle':0.0, 'score':self.score}}   # 

        #print(self.name)
        #print(msgInfluenceTable)
        return msgInfluenceTable


    def broadcast(self):
        """If node j is within node j's radius, record distance; else 0. For influence martrix"""
        for firefly in self.swarm:
            distance =  np.linalg.norm(self.X-firefly.X)
            if distance <= self.transmission_range: # Changed from self.score
                if firefly.name != self.name:
                    errorPoint = locating_error(self.X,firefly.X, self.locationErrorRate)
                    polar_cor = to_relative_polar(self.X, errorPoint)
                    msgInfluenceTable = self.prepareMsgToSend()
                    # msgInfluenceTable = self.influenceTable
                    firefly.receive(self.name, msgInfluenceTable, polar_cor) 


    def receive(self, node_name, node_msgInfluenceTable, polar_cor):
        """update the influenceTable"""
        for key, value in node_msgInfluenceTable.items():
            tableEntry_name = key
            tableEntry_distance = value['distance']
            tableEntry_score = value['score']
            tableEntry_angle = value['angle']
            tableEntry_polar = np.array([tableEntry_distance,tableEntry_angle])
            tableEntrytoSelf = to_relative_polar(to_cartesian(tableEntry_polar,np.array([0,0])) + to_cartesian(polar_cor,np.array([0,0])), np.array([0,0]))
            if (tableEntry_name not in self.influenceTable) or (self.influenceTable[tableEntry_name]['score'] < tableEntry_score):
                self.influenceTable[tableEntry_name] = {'distance':tableEntrytoSelf[0], 'angle':tableEntrytoSelf[1], 'score':tableEntry_score}
            

    def nextWaypoint(self): 
        """Calculate the node's next waypoint based on the Table of influences."""
        """Choose the best node"""
        new_position = np.array([0.0, 0.0])
        potentialnodeList = []
        scoreDiffList = []
        for firefly_name in self.influenceTable:
            firefly_score = self.influenceTable[firefly_name]['score']
            firefly_angle = self.influenceTable[firefly_name]['angle']
            firefly_distance = self.influenceTable[firefly_name]['distance']
            weaken_factor = int(firefly_distance/self.transmission_range)
            firefly_score = firefly_score*(gamma**weaken_factor) 
            if firefly_distance != 0 and self.score < firefly_score:
                scoreDiffList.append(firefly_score - self.score)
                potentialnodeList.append([firefly_distance,firefly_angle])

        if len(scoreDiffList) != 0:
            # # Make a weighted random choice of node
            # total = sum(scoreDiffList)
            # probabilities = [element / total for element in scoreDiffList]
            # chosennode = weighted_random_choice(potentialnodeList, probabilities)
            # new_position = to_cartesian(chosennode, self.X)

            # Make choice only the best one
            chosennode = potentialnodeList[scoreDiffList.index(max(scoreDiffList))]
            new_position = to_cartesian(chosennode, self.X)
            
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
            # Compute firefly logic
            self.sensing()
            if i%self.broadcast_cycle == 0:
                self.broadcast()
            self.move()

            # Store positions for later analysis or plotting
            if i % 10 == 0:    
                self.positions.append(self.X.copy()) # take the records of position
                #print("name: ", self.name, " - score: ", self.score)
            i = i + 1

            # Advance simulated time by 1 time unit
            yield self.env.timeout(1)