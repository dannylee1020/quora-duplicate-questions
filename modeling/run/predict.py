import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report

from run import config

def make_prediction(model_path, data):
	model = load_model(model_path)
	y_pred = model.predict(data)

	return y_pred


if __name__ == '__main__':

	q1_test = np.load(open(config.DATA_DIR + config.Q1_TEST_FILE, 'rb'))
	q2_test = np.load(open(config.DATA_DIR + config.Q2_TEST_FILE, 'rb'))
	y_test = np.load(open(config.DATA_DIR + config.TEST_TARGET_FILE, 'rb'))

	prediction = make_prediction(config.MODEL_DIR + config.MODEL_PATH, [q1_test, q2_test])
	prediction = np.round(prediction)
	report = classification_report(y_test, prediction)

	print(report)



