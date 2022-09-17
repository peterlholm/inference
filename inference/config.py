"inference constants"
from pathlib import Path

MODEL_FOLDER = Path(__file__).parent.parent / 'models'
H_MODEL_FILE = MODEL_FOLDER / 'nn1-0719.h5'
L_MODEL_FILE = MODEL_FOLDER / 'L_model.h5'

HEIGHT_DB_FILE = MODEL_FOLDER / 'DDbase.npy'

#MODEL_FOLDER2 = Path(__file__).parent.parent / 'models/model.dec'
#H_MODEL_FILE = MODEL_FOLDER2 / 'H_model.h5'
#L_MODEL_FILE = MODEL_FOLDER / 'L_model.h5'

#HEIGHT_DB_FILE = MODEL_FOLDER.parent / 'DDbase.npy'

HEIGHT = 160
WIDTH = 160

_DEBUG = True
