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
p.loadURDF("bullet-model/go1_description/urdf/go1.urdf", [0, 0, 0.5])
p.loadURDF("plane.urdf")

# Set up the simulation parameters
p.setGravity(0, 0, -9.81)  # Set gravity
timeStep = 1.0 / 500.0  # Set time step

# Run the simulation loop
while True:
    p.stepSimulation()  # Step the simulation
    time.sleep(timeStep)  # Sleep to control the simulation speed
