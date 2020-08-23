import numpy as np
import nltk
nltk.download('popular')

from nltk import word_tokenize
from keras.models import load_model


def clean_question(question):
	tokens = word_tokenize(question)
	tokens = [t for t in tokens if t.isalpha()]
	tokens = ' '.join(tokens)
	return tokens



def process_question(question):
	clean_q = []
	for q in question:
		q = str(q)
		q = clean_question(q)
		clean_q.append(q)
	return clean_q



def make_prediction(model_path, data):
	model = load_model(model_path)
	y_pred = model.predict(data)

	return y_pred
