import pybullet as p
import pybullet_data
import time

# Initialize PyBullet and load the robot model
p.connect(p.GUI)  # Connect to the GUI
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Set the search path for assets
# Set camera for better visualization
p.resetDebugVisualizerCamera(
    cameraDistance=1.5,  # Distance from the target
    cameraYaw=50,  # Horizontal rotation
    cameraPitch=-35,  # Vertical rotation
    cameraTargetPosition=[0, 0, 0.3],  # Camera target
)

# Load the robot model
quadrupedRobotId = p.loadURDF("bullet-model/go1_description/urdf/go1.urdf", [0, 0, 0.5])
planeId = p.loadURDF("plane.urdf")

# Set up the simulation parameters
p.setGravity(0, 0, -9.81)  # Set gravity
timeStep = 1.0 / 500.0  # Set time step

# Get the number of joints in the robot
numJoints = p.getNumJoints(quadrupedRobotId)

# Enable real-time simulation
p.setRealTimeSimulation(1)

jointIndex = [0, 1, 2]
targetPosition = [0, 0.9, -1.8]

# Run the simulation loop
while True:
    for i in range(3):
        p.setJointMotorControl2(quadrupedRobotId, jointIndex[i], p.POSITION_CONTROL, targetPosition=targetPosition[i])

    p.stepSimulation()  # Step the simulation

    time.sleep(timeStep)  # Sleep to control the simulation speed
