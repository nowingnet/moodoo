# start.py
import sys

sys.path.append('/mnt/disks/sdb/aiffel/moodoo/face_model/train')

import train

if __name__ == "__main__":
    config = '../data/person_1/person_1_config.yml'
    load_checkpoint = ''
    sys.exit(train(config, load_checkpoint, debug=True))