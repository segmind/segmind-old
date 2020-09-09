import os

_edge_tracking_folder = os.path.join(
    os.path.expanduser('~'), '.segmind', 'tracking')
os.makedirs(_edge_tracking_folder, exist_ok=True)

_ROOT_DIR = _edge_tracking_folder
