import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Custom name of the run. Note that all runs are saved in the 'models' directory and have the training time prefixed.",
    )
    parser.add_argument(
        "--num_parallel_envs",
        type=int,
        default=2,
        help="Number of parallel environments while training",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model (.zip). If passed for training, the model is used as the starting point for training. If passed for testing, the model is used for inference.",
    )
    parser.add_argument(
        "--ctrl_type",
        type=str,
        choices=["torque", "position"],
        default="torque",
        help="Whether the model should control the robot using torque or position control.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--lr_p",
        type=float,
        default=3e-4,
        help="Learning rate for Policy Net",
    )
    parser.add_argument(
        "--lr_v",
        type=float,
        default=3e-4,
        help="Learning rate for Value Net",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4000,
    )
    parser.add_argument(
        "--mini_batch_size",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--ppo_epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Maximum iterations to run",
    )
    parser.add_argument(
        "--eval_iter",
        type=int,
        default=50,
        help="Iterations to evaluate the model",
    )
    parser.add_argument(
        "--save_iter",
        type=int,
        default=100,
        help="Iterations to save the model",
    )
    args = parser.parse_args()
    return args
