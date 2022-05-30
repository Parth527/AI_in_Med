import pickle
from data_loader_parth import DataLoader
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

AF_DATASET_DIR = '' # give folder path containing .mat files
LABEL_PATH = '\REFERENCE.csv' # give csv path


dataloader = DataLoader()

ECG_AF = dataloader.load_af_challenge_db(AF_DATASET_DIR, LABEL_PATH, save=True)
AF_hrbt = dataloader.process_signals(signals=ECG_AF, sampling_rate=300, save=True, save_name='AF_heartbeat.pkl')
X_af, y_af = dataloader.prepare_input_challenge(AF_hrbt, save=True)
model = load_model("1DCNN_best_model.h5")
y_pred = model.predict(X_af)
y = y_pred.argmax(axis = 1)
print(classification_report(y_af, y))