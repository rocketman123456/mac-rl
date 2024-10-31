import mujoco
import mujoco.viewer
import time
import threading


# Load the model
model = mujoco.MjModel.from_xml_path("mujoco-model/unitree_go1/scene.xml")
data = mujoco.MjData(model)
viewer = mujoco.viewer.launch(model, data)


def sync_loop():
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.010)


thread = threading.Thread(target=sync_loop)
thread.start()
# Run a viewer to interact with the simulation
while viewer.is_running():
    mujoco.mj_step(model, data)
    # mujoco.mj_forward(model, data)

thread.join()
