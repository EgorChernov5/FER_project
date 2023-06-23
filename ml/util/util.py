from pathlib import Path
import json


REPO_DIR = Path(__file__).parent.parent


def open_json(path):
    data = None
    with open(path, 'r') as json_file:
        data = json.load(json_file)
    return data


def save_as_json(path, data):
    with open(path, "w") as json_file:
        json.dump(data, json_file)
