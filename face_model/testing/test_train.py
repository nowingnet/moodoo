# start.py
import os
import sys

if __name__ == "__main__":
    project_path = '/mnt/disks/sdb/aiffel/moodoo/face_model/'
    os.chdir(project_path)
    sys.path.append(os.getcwd())
    from train import train

    config = 'data/person_1/person_1_config.yml'
    load_checkpoint = ''
    train.train(config, load_checkpoint, debug=True)    