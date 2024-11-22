from framework.utils.utils import *
import os

MODEL_DIR = "logs/models"
LOG_DIR = "logs/logs"

if __name__ == "__main__":
    args = parse_args()
    print(args)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    # train(args)
