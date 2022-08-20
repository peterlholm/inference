"inference constants"
from pathlib import Path

MODEL_FOLDER = Path(__file__).parent.parent / 'models'
print(MODEL_FOLDER)
H_MODEL_FILE = MODEL_FOLDER / 'nn1-0719.h5'
L_MODEL_FILE = MODEL_FOLDER / 'L_model.h5'

HEIGHT_DB_FILE = MODEL_FOLDER / 'DDbase.npy'

HEIGHT = 160
WIDTH = 160
