import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
DATA_DIR = os.path.join(ROOT_DIR, "data")
FIGURE_DIR = os.path.join(ROOT_DIR, "figures")
MAP_DIR = os.path.join(FIGURE_DIR, "maps")
MODEL_DIR = os.path.join(ROOT_DIR, "models")


def init_data_folder(abs_folder_path: str):
    if not os.path.exists(abs_folder_path):
        os.makedirs(abs_folder_path, exist_ok=True)
