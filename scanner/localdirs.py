import os
import appdirs

data_dir = os.path.join(appdirs.user_data_dir(), "lettucethink/scanner")
scanners_dir = os.path.join(data_dir, "scanners")
objects_dir = os.path.join(data_dir, "objects")
paths_dir = os.path.join(data_dir, "paths")
db_dir = os.path.join(data_dir, "data")

def create_directories():
    if not os.path.isdir(scanners_dir):
        os.makedirs(scanners_dir)
    if not os.path.isdir(objects_dir):
        os.makedirs(objects_dir)
    if not os.path.isdir(paths_dir):
        os.makedirs(paths_dir)
    if not os.path.isdir(db_dir):
        os.makedirs(db_dir)
