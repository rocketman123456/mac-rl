import pybullet as p
import pybullet_data
import numpy as np
import time
from multiprocessing import Pool


# Function to initialize and simulate a single robot
def simulate_robot(robot_id):
    # Connect to PyBullet
    if robot_id == 0:
        physicsClient = p.connect(p.GUI)
    else:
        physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)

    # Load a simple robot model (a box in this case)
    robot_pos = [np.random.uniform(-5, 5), np.random.uniform(-5, 5), 0.5]
    robot = p.loadURDF("model/bullet/go1_description/urdf/go1.urdf", basePosition=robot_pos)  # Load ground
    # box = p.loadURDF("cube.urdf", basePosition=robot_pos)  # Create a box as the robot
    plane = p.loadURDF("plane.urdf")

    # Simulate for a number of steps
    for step in range(100):  # Simulate 100 time steps
        p.stepSimulation()
        # Example movement: move robot randomly
        p.applyExternalForce(robot, -1, [np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0], [0, 0, 0], p.WORLD_FRAME)
        time.sleep(0.01)  # Sleep to visualize if needed

    p.disconnect()


if __name__ == "__main__":
    num_robots = 1000  # Number of robots to simulate
    robot_ids = range(num_robots)

    # Using multiprocessing to simulate robots in parallel
    with Pool() as pool:
        pool.map(simulate_robot, robot_ids)

    print("Simulation complete.")
