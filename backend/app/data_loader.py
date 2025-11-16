import os
import json
import numpy as np

BASE_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data")

def list_all_episodes():
    episodes = []

    for main_folder in os.listdir(BASE_DATA_PATH):
        folder_path = os.path.join(BASE_DATA_PATH, main_folder)
        if not os.path.isdir(folder_path):
            continue
        
        # Parse folder name → e.g., AF_Approved → event=AF, approved=True
        parts = main_folder.split("_")
        event_type = parts[0]
        approved = parts[1].lower() == "approved"

        # Loop episode folders
        for episode_id in os.listdir(folder_path):
            episode_path = os.path.join(folder_path, episode_id)
            if os.path.isdir(episode_path):
                episodes.append({
                    "id": episode_id,
                    "event_type": event_type,
                    "approved": approved,
                    "path": episode_path
                })

    return episodes


def load_episode(episode_path):
    # Read metadata
    json_file = [f for f in os.listdir(episode_path) if f.endswith(".json")][0]
    meta_path = os.path.join(episode_path, json_file)
    metadata = json.load(open(meta_path))

    ecg_files = sorted([f for f in os.listdir(episode_path) if f.endswith(".txt")])
    traces = []

    for file in ecg_files:
        file_path = os.path.join(episode_path, file)
        arr = np.loadtxt(file_path, delimiter=",")  # shape = (6000,2)
        traces.append(arr)

    full_ecg = np.vstack(traces)  # shape = (18000,2)

    return metadata, full_ecg
