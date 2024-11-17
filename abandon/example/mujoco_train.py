import numpy as np
import threading
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from dm_control import suite
from dm_control import viewer

# The basic mujoco wrapper.
from dm_control import mujoco

# Access to enums and MuJoCo library functions.
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib


def display_video(frames, framerate=30):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use("Agg")  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_data(frame)
        # frame = np.array(fr ame)
        return [im]

    interval = 1000 / framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames, interval=interval, blit=True, repeat=False)
    anim.save("example.mp4", writer="ffmpeg", fps=30)
    # plt.show()
    # return HTML(anim.to_html5_video())


# Function to run an environment
def run_env(domain_name, task_name, id):
    framerate = 30  # (Hz)

    # Visualize the joint axis
    scene_option = mujoco.wrapper.core.MjvOption()
    scene_option.flags[enums.mjtVisFlag.mjVIS_JOINT] = True

    env = suite.load(domain_name=domain_name, task_name=task_name)
    time_step = env.reset()

    # Simulate and display video.
    # if id == -1:
    #     frames = []

    while True:
        action = np.random.uniform(env.action_spec().minimum, env.action_spec().maximum)
        time_step = env.step(action)

        # print(f"Env: {domain_name}, Task: {task_name}, Observation: {time_step.observation}")
        # print("Reward:", time_step.reward)
        # if id == -1:
        #     pixels = env.physics.render(scene_option=scene_option)
        #     frames.append(pixels)

        if time_step.last():
            break

    # if id == -1:
    #     display_video(frames, framerate)
    # viewer_instance.close()


# Define the environments to run
environments = [
    ("humanoid", "walk"),
    # ("humanoid", "walk"),
    # ("humanoid", "walk"),
    # ("humanoid", "walk"),
]

# Create and start threads
threads = []
env_id = 0
for domain_name, task_name in environments:
    # run_env(domain_name, task_name, env_id)
    thread = threading.Thread(target=run_env, args=(domain_name, task_name, env_id))
    threads.append(thread)
    env_id = env_id + 1
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()
