from process import config
from process.predict import make_prediction
import numpy as np


def test_single_output():
	# Given
	q1_test = np.load(open(config.DATA_DIR + config.Q1_TEST_FILE, 'rb'))[:1]
	q2_test = np.load(open(config.DATA_DIR + config.Q2_TEST_FILE, 'rb'))[:1]

	# When 
	y_pred = make_prediction(config.MODEL_DIR + config.MODEL_PATH, [q1_test, q2_test])

	# Then 
	assert len(y_pred) is not None



def test_multiple_outputs():
	# Given
	q1_test = np.load(open(config.DATA_DIR + config.Q1_TEST_FILE, 'rb'))
	q2_test = np.load(open(config.DATA_DIR + config.Q2_TEST_FILE, 'rb'))

	# When
	y_pred = make_prediction(config.MODEL_DIR + config.MODEL_PATH, [q1_test, q2_test])

	# Then
	assert len(y_pred) == len(q1_test)



