import numpy as np
import json
import matplotlib.pyplot as plt
import itertools
import heapq
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
import sys
import numpy as np
from scipy.ndimage import label
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from ros2_aruco_interfaces.msg import ArucoMarkers
from geometry_msgs.msg import Twist
import time
import uuid
from action_msgs.srv import CancelGoal
from actionlib_msgs.msg import GoalID
from scipy.spatial.transform import Rotation as R
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf_transformations import quaternion_matrix
import tf_transformations
import tf2_ros 
import json
from std_msgs.msg import Int32MultiArray


from tf2_geometry_msgs import do_transform_pose

class pathSolver:
    def toGridCoord(self, dest, origin):
        gridU = (int)((dest[0]-origin[0])/self.resolution)
        gridV = (int)((dest[1]-origin[1])/self.resolution)
        return (gridU, gridV)
    def existFreePath(self, coord):
        for i in range(-1, 2,1):
            for j in range(-1,2,1):
                if i==0 and j==0:
                    continue
                index = (coord[0]+i, coord[1]+j)
                if(self.grid[index]==0):
                    return True
        return False   
    # Euclidian Distance Heuristic
    def heuristic(self, a, b):
        return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

    def astar(self, start, goal):

        neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]

        # Save the shortest path -> Save on a given from which neighbor we come from
        came_from = {}
        # g[n], cheapest path from start to n
        gscore = {start:0}
        # Guess of best path from start to finish, going through n
        # f[n] = g[n] + h[n]
        fscore = {start:self.heuristic(start, goal)}
        oheap = []
        # Minimum Priority Queue
        heapq.heappush(oheap, (fscore[start], start))
        
        while oheap:

            current = heapq.heappop(oheap)[1]
            # If current case is goal -> Return total distance and total path
            if current == goal:
                data = []
                distance = 0.0
                while current in came_from:
                    data.append(current)
                    distance += self.heuristic(current, came_from[current])
                    current = came_from[current]
                return data, distance

            #close_set.add(dists[index]current)

            for i, j in neighbors:
                neighbor = (current[0] + i, current[1] + j) 
                # Check that neighbor is in the grid and is not an obstacle
                if not 0 <= neighbor[0] <= self.grid.shape[0] or not 0 <= neighbor[1] <= self.grid.shape[1]:
                    continue 
                if self.grid[neighbor] == 1.0 or self.grid[neighbor] == -1.0:
                    continue
                # Compute new heuristic Score of neighbor       
                tentative_g_score = gscore[current] + self.heuristic(current, neighbor)
                # If this is the smallest heuristic score then the path from start to end passing by neighbor is minimized by passing through current
                if  tentative_g_score < gscore.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    if neighbor not in [i[1]for i in oheap]:
                        heapq.heappush(oheap, (fscore[neighbor], neighbor))    
        return False, 0.0   
    def computeDist(self):
        posList = [self.posRobot] + self.posArucos
        pathDict={}
        distDict={}
        for i in range(len(posList)):
            for j in range(len(posList)):
                index = (i,j)
                pathDict[index], distDict[index] = self.astar(posList[i],posList[j])
        return pathDict, distDict      
    def nn_salesman(self):
        nodes = list(range(max(list(self.dists.keys()))[0]+1)) # Createe [0 ... #Arucos], 0 is the position of the robot
        sizeNodes = len(nodes)
        visited = set()
        current_node = 0 
        path = []
        total_distance = 0

        while len(visited) < sizeNodes-1:
            nearest_distance = float('inf')
            nearest_node = None

            for neighbor in nodes:
                if neighbor != current_node and neighbor not in visited:
                    distance = self.dists[(current_node, neighbor)]
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_node = neighbor

            path.append((current_node, nearest_node))
            total_distance += nearest_distance
            visited.add(current_node)
            current_node = nearest_node
        return total_distance, path    
    def __init__(self, occupancyGridFile, positionFile, resolution = 0.05):
        # Open Files
        self.resolution = resolution
        with open(occupancyGridFile, 'r') as f:
            grid = [[float(num) for num in line.split()] for line in f]
        self.grid = np.array(grid)
        with open(positionFile, "r") as posFile:
            positionData = json.load(posFile)
        # Preprocess the files
        origin = (0,0)
        pos_robot = (0,0)
        pos_arucos= []
        nav2PosRobot = (0,0)
        nav2PosArucos = []
        for item in positionData:
            index = (item[1][1], item[1][0])
            if(item[0] == "Origin Position"):
                origin = index
            elif(item[0] == "Current Position of the robot"):
                nav2PosRobot = (item[1][0], item[1][1])
                pos_robot = index
            else:
                pos_arucos.append(index) 
                nav2PosArucos.append((item[1][0], item[1][1]))  
        # Position of Robot and Arucos
        self.posRobot = self.toGridCoord(pos_robot, origin)
        self.posArucos = [self.toGridCoord(element, origin) for element in pos_arucos]
        indicesFreeArucos = [i for i, pos in enumerate(self.posArucos) if self.existFreePath(pos)]
        self.posArucos = [self.posArucos[i] for i in indicesFreeArucos]
        nav2PosArucos = [nav2PosArucos[i] for i in indicesFreeArucos]
        #Create dictionnary mapping nodes <-> Nav2 positions
        self.nodeDict = {index+1: nav2Pos for index, nav2Pos in enumerate(nav2PosArucos)}
        self.nodeDict[0] = nav2PosRobot
        print(self.nodeDict)
        # Preprocess the Grid
        self.grid[self.posRobot] = 2.0
        for indices in self.posArucos:
            self.grid[indices] = 3.0
        #Compute Distances:
        self.paths, self.dists = self.computeDist()
        # NN Method
        self.total_distance, self.path = self.nn_salesman()
    def __iter__(self):
        self.it = -1
        return self
    def __next__(self):
        self.it+=1
        if self.it < len(self.path):
            x,y = self.nodeDict[self.path[self.it][1]]
            return x, y
        raise StopIteration
    def display_grid(self, grid=None):
        if grid is None:
            grid=self.grid
        fig, ax = plt.subplots(figsize=(12, 12))
        cmap = plt.get_cmap('tab10')
        colors = {1.0: cmap(0), 0.0: cmap(1), -1.0 : cmap(2), 2.0 : cmap(3), 3.0 : cmap(4), 4.0 : cmap(5), 5.0 : cmap(6)}
        ax.matshow([[colors[j] for j in i] for i in grid])
        plt.show()
    def display_solution(self):
        gridPathNN = self.grid.copy()
        totalPathNN = []
        for edge in self.path:
            totalPathNN += paths[edge]
        for indices in totalPathNN:
            gridPathNN[indices]=5.0
        self.display_grid(gridPathNN)
    def getDistsInfos(self):
        return self.paths, self.dists
    def getOptimalPath(self):
        return self.path

class robotActionClient(Node):
	def __init__(self):#constructor
		super().__init__('robot_action_client')#calls the super constructor
		
		self._action_client=ActionClient(self, NavigateToPose,'navigate_to_pose')#requires the node to add the action client, the action type that we require to do something, the action name
		#Our action client will be able to communicate with action servers of the same action name and type.
		
		self.subscription = self.create_subscription(OccupancyGrid,'map',self.listener_callback,10)
		self.aruco_subscription = self.create_subscription(
		ArucoMarkers,  # Message type
		'aruco_markers',  # Topic name
		self.aruco_callback,  # Callback functio
		10  # QoS profile, adjust as n/aruco_markers
		)	
		self.delivery_subscription = self.create_subscription(
		Int32MultiArray,
		'/delivery_locations',
		self.delivery_callback,
		10)
		self.tf_buffer = tf2_ros.Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)
		self.cmd_vel_publisher= self.create_publisher(Twist, '/cmd_vel', 10)	
		self.finishFlag=1
		self.get_logger().info('Node instantiated')
		# Tuning for Rotation
		self.aruco_id=[]
		self.aruco_seen=[]
		self.aruco_positions=[]
		self.current_position_x=0.
		self.current_position_y=0.
		
		self.visited_positions=[]
		self.phase_mapping=True
		self.number_arucos=10
		self.start=True
		self.delivering=False
		self.counter=0
		self.received=False
		self.rotating=False
		self.stopRotation=False
	def delivery_callback(self,msg):
		if not self.received:
			self.aruco_to_visit=msg.data
			self.received=True
		else:
			return 
	def aruco_callback(self, msg):
		if not self.tf_buffer.can_transform('camera_optical_joint', 'camera_link', tf2_ros.Time()) or \
		not self.tf_buffer.can_transform('camera_link', 'base_link', tf2_ros.Time()) or \
		not self.tf_buffer.can_transform('base_link', 'base_footprint', tf2_ros.Time()) or \
		not self.tf_buffer.can_transform('base_footprint', 'odom', tf2_ros.Time()) or \
		not self.tf_buffer.can_transform('odom', 'map', tf2_ros.Time()):
			print('I disregard')
			return

		for index in range(len(msg.marker_ids)):
			aruco_id = msg.marker_ids[index]
			transform = None

			try:
				transform = self.tf_buffer.lookup_transform('map', 'camera_optical_joint', rclpy.time.Time())
				aruco_transformed = do_transform_pose(msg.poses[index], transform)
			except tf2_ros.ExtrapolationException as e:
				print("Transform not available: {}".format(e))
				continue

			orientation_quaternion = [
			aruco_transformed.orientation.x,
			aruco_transformed.orientation.y,
			aruco_transformed.orientation.z,
			aruco_transformed.orientation.w
			]
			angleX, angleY, angleZ = tf_transformations.euler_from_quaternion(orientation_quaternion)
			arucoPoseForRobotX = aruco_transformed.position.x - np.cos(angleZ + (np.pi / 2)) * 0.5  # record position 30 cm away for robot
			arucoPoseForRobotY = aruco_transformed.position.y - np.sin(angleZ + (np.pi / 2)) * 0.5
			aruco_orientation = (angleZ + np.pi / 2) % (2 * np.pi)

			if aruco_id in self.aruco_id:
				index_existing = self.aruco_id.index(aruco_id)
				old_position, old_orientation, count = self.aruco_positions[index_existing]

				# Update the count of sightings
				new_count = count + 1

				# Calculate the new averaged position and orientation
				new_position_x = (old_position[0] * count + arucoPoseForRobotX) / new_count
				new_position_y = (old_position[1] * count + arucoPoseForRobotY) / new_count
				new_position_z = (old_position[2] * count + aruco_transformed.position.z) / new_count
				new_angleX = (old_orientation[0] * count + angleX) / new_count
				new_angleY = (old_orientation[1] * count + angleY) / new_count
				new_angleZ = (old_orientation[2] * count + aruco_orientation) / new_count

				# Update the stored position, orientation, and count
				self.aruco_positions[index_existing] = (
				(new_position_x, new_position_y, new_position_z),
				(new_angleX, new_angleY, new_angleZ),
				new_count
				)

				print('Aruco_id:', aruco_id, 'déjà vu - Updated position')
			else:
				if self.rotating:
					self.stopRotation=True
					time.sleep(0.5)#remove it if you want
					
					return 
				self.aruco_id.append(aruco_id)
				self.aruco_positions.append((
				(arucoPoseForRobotX, arucoPoseForRobotY, aruco_transformed.position.z),
				(angleX, angleY, aruco_orientation),
				1  # Initial count
				))
				print('I just spotted Aruco number:', aruco_id)
				print('Aruco Position:', arucoPoseForRobotX, arucoPoseForRobotY)

	def find_arucos(self,MyOccupancyGrid):#check for indices
		self.display_arucos()
		with open('output.txt', 'w') as file:
			for row in MyOccupancyGrid:
		# Join elements of the row with spaces and write to file
				file.write(' '.join(map(str, row)) + '\n')
		MyOccupancyGrid=self.preprocessing(MyOccupancyGrid)
		with open('outputProcessed.txt', 'w') as file:
			for row in MyOccupancyGrid:
		# Join elements of the row with spaces and write to file
				file.write(' '.join(map(str, row)) + '\n')

		positions_to_save=[]
		x=self.current_position_x
		y=self.current_position_y
		positions_to_save.append(('Origin Position', (self.originX, self.originY)))
		positions_to_save.append(('Current Position of the robot', (x, y)))

		for i in range(len(self.aruco_id)):
			if (self.aruco_id[i] in self.aruco_to_visit):
				positions_to_save.append((self.aruco_id[i], (self.aruco_positions[i][0][0], self.aruco_positions[i][0][1])))

		with open('positions.txt', 'w') as file:
			json.dump(positions_to_save, file, indent=4)
		print(len(self.aruco_id))
		if(len(self.aruco_id)>=self.number_arucos):
			print('je passe')
			self.delivering=True
			self.pizza_path()
			return 
		potentialGoal=self.create_potential_goal(MyOccupancyGrid)#find the potential place to o
		findNext=potentialGoal.copy()
		for index in range(0, len(self.visited_positions),1):
			xIndex=(self.visited_positions[index][0]-self.originX)/self.mapResolution
			yIndex=(self.visited_positions[index][1]-self.originY)/self.mapResolution
			findNext[int(yIndex)][int(xIndex)]=2#these elements are the ones already visited 
		indices=self.find_furthest_index(findNext)
		indexX=indices[1]
		indexY=indices[0]

		self.goalX = (indexX * self.mapResolution) + self.originX
		self.goalY = (indexY* self.mapResolution) + self.originY#ptt inverser ici
		self.visited_positions.append((self.goalX,self.goalY))
		print('Position x à atteindre: ',self.goalX)
		print('Position y à atteindre: ',self.goalY)
		self.send_goal()



	def find_furthest_index(self,matrix):#finds the one lemenjt the most distance from all the two 
	# Find indices of elements equal to 1 and 2
		positions_to_save=[]
		x=self.current_position_x
		y=self.current_position_y
		positions_to_save.append(('Origin Position', (self.originX, self.originY)))
		positions_to_save.append(('Current Position of the robot', (x, y)))

		for i in range(len(self.aruco_id)):
			if (self.aruco_id[i] in self.aruco_to_visit):
				positions_to_save.append((self.aruco_id[i], (self.aruco_positions[i][0][0], self.aruco_positions[i][0][1])))

		with open('positions.txt', 'w') as file:
			json.dump(positions_to_save, file, indent=4)
		one_indices = np.argwhere(matrix == 1)
		two_indices = np.argwhere(matrix == 2)

		max_distance = -1
		furthest_index = None

		# Iterate over each index equal to 1
		for one_index in one_indices:
			# Calculate distance from each index equal to 2
			distances = np.linalg.norm(two_indices - one_index, axis=1)
			min_distance_to_two = np.min(distances)

			# Update furthest index if the distance is greater than max_distance
			if min_distance_to_two > max_distance:
				max_distance = min_distance_to_two
				furthest_index = tuple(one_index)

		return furthest_index

	def pizza_path(self):
		print('Calculating path')
		myPathSolver = pathSolver('outputProcessed.txt', 'positions.txt')
		self.finishFlag=False
		self.list=[(x,y) for x,y in myPathSolver]
		min_distance = float('inf')
		print('je sors de pizza_path')
		self.send_next_aruco()
	
	def send_next_aruco(self):
		if self.counter>=(len(self.list)):
			sys.exit()
		self.goalX=self.list[self.counter][0]
		self.goalY=self.list[self.counter][1]
		self.counter+=1
		min_distance = float('inf')
		for i in range(0,len(self.aruco_positions),1):
			dist=distance_2points(self.goalX,self.goalY,self.aruco_positions[i][0][0],self.aruco_positions[i][0][1])
			if dist<min_distance:
				min_distance=dist
				index=i
				
		self.index=index
		print('va envoyer new goal')
		self.send_goal()
	def display_arucos(self):
		# Function to display all detected Aruco markers
		print("Detected Aruco markers:")
		for index in range(0,len(self.aruco_positions),1):
		    print("Aruco ID", self.aruco_id[index])
		    print("Aruco x Position", self.aruco_positions[index][0][0])
		    print("Aruco y Position", self.aruco_positions[index][0][1])
		    print("Aruco z Position", self.aruco_positions[index][0][2])
			
	def rotate_full_circle(self):
	# Faire un tour complet sur lui-même
        self.rotating=True
		twist = Twist()
		twist.angular.z = 0.3  # Vitesse de rotation
		start_time = time.time()
		while time.time() - start_time < 2 * 3.14/twist.angular.z :  # Faire une rotation complète (2 * pi / vitesse de rotation)
		    self.cmd_vel_publisher.publish(twist)
			if self.stopRotation:#one new aruco is found
				self.stopRotation=False
				break
		twist.angular.z = 0.  # Arrêter la rotation
		self.cmd_vel_publisher.publish(twist)
		self.rotating=False
		print('rotation over')


	def listener_callback(self, msg):#this function is called each time a occupancyGrid message is received
		if self.finishFlag==1 and self.delivering==False:#the goal is reached, I want to get the map to take the next decision
			self.rotate_full_circle()
			print(self.delivering)
			self.finishFlag=0
			self.get_logger().info('"%s"' % msg.info.width)
			self.get_logger().info('"%s"' % msg.info.height)
			self.get_logger().info('"%s"' % msg.info.origin.position.x)
			self.get_logger().info('"%s"' % msg.info.origin.position.y)
			self.get_logger().info('"%s"' % msg.info.resolution)
			self.get_logger().info('Position du robot:' )
			self.get_logger().info('"%s"' % self.current_position_x)
			self.get_logger().info('"%s"' % self.current_position_y)
			self.mapResolution=msg.info.resolution
			self.originX= msg.info.origin.position.x
			self.originY= msg.info.origin.position.y
			mapX=(int)(0.-msg.info.origin.position.x)/msg.info.resolution
			mapY=(int)(0.-msg.info.origin.position.y)/msg.info.resolution

			currentX=(int)(self.current_position_x-msg.info.origin.position.x)/msg.info.resolution
			currentY=(int)(self.current_position_x-msg.info.origin.position.y)/msg.info.resolution
			print('Indice x de l origine',mapX)
			print('Indice y de l origine',mapY)
			print('Indice x du robot', currentX)
			print('Indice y du robot', currentY)
			MyOccupancyGrid=np.zeros((msg.info.height, msg.info.width))

			for i in range (msg.info.height):#extract from the serialized data,The map data, in row-major order, starting with (0,0).
				for j in range (msg.info.width):
					MyOccupancyGrid [i][j] = msg.data[i * msg.info.width + j]
			if self.phase_mapping==True:
				
				self.searchForNextGoal(MyOccupancyGrid)
			else:
				self.find_arucos(MyOccupancyGrid)

		elif self.finishFlag==1 and self.delivering==True:
			print('I stopped in front of aruco')
			self.finishFlag=0
			time.sleep(5)
			self.send_next_aruco()
	def searchForNextGoal(self,MyOccupancyGrid):
		with open('output.txt', 'w') as file:
			for row in MyOccupancyGrid:
		# Join elements of the row with spaces and write to file
				file.write(' '.join(map(str, row)) + '\n')
		MyOccupancyGrid=self.preprocessing(MyOccupancyGrid)
		with open('outputProcessed.txt', 'w') as file:
			for row in MyOccupancyGrid:
		# Join elements of the row with spaces and write to file
				file.write(' '.join(map(str, row)) + '\n')
		potentialGoal=self.create_potential_goal(MyOccupancyGrid)
		frontiers=self.searchFrontiers(MyOccupancyGrid)
		cluster_position=self.middle_of_closest_cluster(frontiers)
		closest_position=self.get_final_goal(potentialGoal,cluster_position[0],cluster_position[1])
		self.goalX=closest_position[0]
		self.goalY=closest_position[1]
		print('Position x actuelle: ',self.current_position_x)
		print('Position y actuelle: ',self.current_position_y)
		print('Position x à atteindre idéalement : ',cluster_position[0])
		print('Position y à atteindre idéalement: ',cluster_position[1])
		print('Position x à atteindre: ',self.goalX)
		print('Position y à atteindre: ',self.goalY)
		
		with open('outputFrontier.txt', 'w') as file:
			for row in frontiers:
		# Join elements of the row with spaces and write to file
				file.write(' '.join(map(str, row)) + '\n')
		with open('outputgoal.txt', 'w') as file:
			for row in potentialGoal:
		# Join elements of the row with spaces and write to file
				file.write(' '.join(map(str, row)) + '\n')
		self.send_goal()
		
	def searchFrontiers(self,array):
		result=np.zeros_like(array)
		# Parcourir le tableau d'entrée
		for i in range(len(array)):
			for j in range(len(array[i])):
			# Si la cellule contient un 0
				if array[i][j] == 0:
				# Vérifier les voisins (haut, bas, gauche, droite)
					for x, y in [(i-1, j), (i+1, j), (i, j-1), (i, j+1),(i-1,j-1),(i+1,j-1),(i-1,j+1),(i+1,j+1)]:
						# Vérifier si les indices sont valides et si le voisin est -1
						if 0 <= x < len(array) and 0 <= y < len(array[0]) and array[x][y] == -1:
						# Si un voisin est -1, mettre 1 dans le résultat et passer à la prochaine cellule
							result[i][j] = 1
							break
		array_np = np.array(result)
		labeled_array, num_clusters = label(array_np)
		return labeled_array
	
	def get_final_goal(self, array,x_position,y_position):
		
		min_distance = float('inf')
		for i in range(0,len(array),1):
			for j in range(0,len(array[i]),1):
				if array[i][j]==1:
					position_x = (j * self.mapResolution) + self.originX
					position_y = (i* self.mapResolution) + self.originY#ptt inverser ici
					distance=np.sqrt((position_x-x_position)**2+(position_y-y_position)**2)
				else:
					continue
				if distance<min_distance:
					best_x=position_x
					best_y=position_y
					min_distance=distance
		return (best_x,best_y)
	def create_potential_goal(self,array):
	# Créer une matrice vide pour stocker le résultat
		potential_goal = np.zeros_like(array)
		# Parcourir chaque élément de array
		for i in range(0,len(potential_goal),1):
			for j in range(0,len(potential_goal[i]),1):
				
				# Vérifier si l'élément est égal à 0 dans array
				if array[i, j] == 0:
				# Vérifier si l'élément a un voisin égal à -1 dans array
					if (has_neighbor_in_range(array, i, j, -1,4)):
						continue
					# Vérifier si aucun voisin dans le voisinage de n indices n'est égal à 1 dans array
					if not (has_neighbor_in_range(array, i, j, 1, 8)):
						potential_goal[i, j] = 1


		return potential_goal



	
	def middle_of_closest_cluster(self, clustered_array):
		# Trouver les clusters et leurs tailles
		unique_clusters, cluster_counts = np.unique(clustered_array, return_counts=True)
		# Retirer le cluster 0 (les 0 sont considérés comme le fond)
		unique_clusters = unique_clusters[1:]
		cluster_counts = cluster_counts[1:]

		# Retirer les clusters dont le nombre d'éléments est inférieur à min_cluster_size
		valid_clusters = unique_clusters[cluster_counts >= 7]

		# Convertir l'indice en position pour chaque cluster
		cluster_positions_x = []
		cluster_positions_y = []
		index=1
		cluster_list=[]
		for cluster in valid_clusters:
			indices = np.where(clustered_array == cluster)
			middle_index = (np.median(indices[0]), np.median(indices[1]))
			position_x = (middle_index[1] * self.mapResolution) + self.originX
			position_y = (middle_index[0] * self.mapResolution) + self.originY
			cluster_positions_x.append(position_x)
			cluster_positions_y.append(position_y)
			cluster_list.append(cluster)

		# Trouver le cluster le plus proche de la position actuelle du robot
		min_distance = float('inf')
		closest_position = None
		best_cluster=-1
		for i in range(len(cluster_positions_x)):

			distance = self.evaluate_distance(cluster_positions_x[i], cluster_positions_y[i])
			if distance < min_distance:
				min_distance = distance
				closest_position = (cluster_positions_x[i], cluster_positions_y[i])
				best_cluster=i
		if best_cluster==-1:
			print('Pas de cluster trouvé')
			self.phase_mapping=False
			return (self.current_position_x,self.current_position_y)

		else:
			print('Cluster sélectionné', cluster_list[best_cluster])
		return closest_position

	def evaluate_distance(self,x,y):
		return np.sqrt((x-self.current_position_x)**2+(y-self.current_position_y)**2)

	def preprocessing(self,MyOccupancyGrid):
		threshold=50.
		OccupancyGridNew=MyOccupancyGrid.copy()
		for i in range(1, len(MyOccupancyGrid), 1):
			for j in range(1, len(MyOccupancyGrid[i]), 1):
				element = MyOccupancyGrid[i][j]
				if element == -1.:
					continue  # Continue with the next iteration of the inner loop
				elif MyOccupancyGrid[i+1][j]>=threshold or MyOccupancyGrid[i-1][j]>=threshold or MyOccupancyGrid[i][j+1]>=threshold or MyOccupancyGrid[i][j-1]>=threshold :#avoid points that are most probaby obstacle or just next to some 
					OccupancyGridNew[i][j]=1.
				elif has_neighbor_in_range(MyOccupancyGrid,i,j,-1,1):
					OccupancyGridNew[i][j]=-1.
				else: 
					OccupancyGridNew[i][j]=0.
		return OccupancyGridNew
	def send_goal(self):
		goal_msg = NavigateToPose.Goal()  # Create a goal message instance
		goal_msg.pose.pose.position.x = self.goalX # Set the x position to be reached
		goal_msg.pose.pose.position.y = self.goalY

	# Send Goal to action Server
		if self.delivering==False:
			goal_msg.pose.pose.orientation.w = 1.  # Set the orientation to be reached
		else:
		
			angleX = self.aruco_positions[self.index][1][0]
			angleY = self.aruco_positions[self.index][1][1]
			angleZ = self.aruco_positions[self.index][1][2]

			# Convertir les angles Euler en quaternion
			quaternion = tf_transformations.quaternion_from_euler(angleX, angleY, angleZ)

			# Assigner les valeurs de quaternion au message goal_msg
			goal_msg.pose.pose.orientation.x = quaternion[0]
			goal_msg.pose.pose.orientation.y = quaternion[1]
			goal_msg.pose.pose.orientation.z = quaternion[2]
			goal_msg.pose.pose.orientation.w = quaternion[3]
		self._action_client.wait_for_server()
		self.get_logger().info('Server available, send goal')
		self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
		#self._send_goal_future = self._action_client.send_goal_async(goal_msg)  # Send the goal message
		print('va faire la fonction')
		self._send_goal_future.add_done_callback(self.goal_response_callback)

	
	# Callback of Sending action
	def goal_response_callback(self, future):
		self._goal_handle = future.result()
		print('in response callback')
		if not self._goal_handle.accepted:
			self.get_logger().info('Goal rejected :(')#we directly return 
			return


		self.get_logger().info('Goal accepted :)')

		self._get_result_future = self._goal_handle.get_result_async()#Now that we’ve got a goal handle, we can use it to request the result with the method get_result_async() because there should be a result
		self._get_result_future.add_done_callback(self.get_result_callback)#we have a result, so we want to call a function to deal with it
		
	# Callback of result of action
	def get_result_callback(self, future):
		resultGoal = future.result()
		if resultGoal:
			self.get_logger().info('I made it!')
			self.finishFlag=1
			self.finishFlagRotation=1
			if self.delivering==False:
				self.visited_positions.append((self.goalX,self.goalY))
			else:
				time.sleep(5)
			
		else:
			self.get_logger().info('I failed')
			rclpy.shutdown()


	def feedback_callback(self, feedback_msg):
		feedback = feedback_msg.feedback
		self.current_position_x=feedback.current_pose.pose.position.x
		self.current_position_y=feedback.current_pose.pose.position.y
		if self.start==True:
			self.start=False
			self.visited_positions.append((self.current_position_x,self.current_position_y))
			

def distance_2points(x1,y1,x2,y2):
	return np.sqrt((x1-x2)**2+(y1-y2)**2)
def transform_position(pose_array, transform):
    transformed_poses = []

    for pose in pose_array:
        # Extract translation and rotation from the transform
        trans = transform.transform.translation
        rot = transform.transform.rotation

        # Convert pose orientation to quaternion
        pose_quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]

        # Create a transformation matrix from the translation and rotation
        transform_matrix = tf_transformations.quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
        transform_matrix[0][3] = trans.x
        transform_matrix[1][3] = trans.y
        transform_matrix[2][3] = trans.z

        # Create a 4x1 vector for the pose position
        pose_position = [pose.position.x, pose.position.y, pose.position.z, 1]

        # Transform the pose position
        transformed_position = transform_matrix.dot(pose_position)

        # Transform the pose orientation
        transformed_orientation = tf_transformations.quaternion_multiply([rot.x, rot.y, rot.z, rot.w], pose_quat)

        # Create the new transformed pose
        new_pose = Pose()
        new_pose.position.x = transformed_position[0]
        new_pose.position.y = transformed_position[1]
        new_pose.position.z = transformed_position[2]
        new_pose.orientation.x = transformed_orientation[0]
        new_pose.orientation.y = transformed_orientation[1]
        new_pose.orientation.z = transformed_orientation[2]
        new_pose.orientation.w = transformed_orientation[3]

        transformed_poses.append(new_pose)

    return transformed_poses

def has_neighbor_in_range(array, i, j, value, n):
# Vérifier les voisins dans un voisinage de n indices
	for di in range(-n, n + 1):
		for dj in range(-n, n + 1):
			ni, nj = i + di, j + dj
			if 0 <= ni < array.shape[0] and 0 <= nj < array.shape[1] and array[ni, nj] == value:
				return True
	return False

def has_bigger_neighbor_in_range(array, i, j, value, n):
# Vérifier les voisins dans un voisinage de n indices
	for di in range(-n, n + 1):
		for dj in range(-n, n + 1):
			ni, nj = i + di, j + dj
			if 0 <= ni < array.shape[0] and 0 <= nj < array.shape[1] and array[ni, nj] >= value:
				return True
	return False
def main(args=None):
	rclpy.init(args=args)
	myNode=robotActionClient()
	rclpy.spin(myNode)
	myNode.destroy_node()
	rclpy.shutdown()
if __name__ == '__main__':
	main()

