import streamlit as st
from keras.preprocessing.sequence import pad_sequences
import pickle
import time
import numpy as np

import config
from process import process_question, clean_question, make_prediction




with open(config.FILE_DIR + config.TOKENIZER, 'rb') as handle:
	tokenizer = pickle.load(handle)


def tokenize_pad_questions(question_1, question_2):
	question_1 = process_question(question_1)
	question_2 = process_question(question_2)

	question1_word_sequence = tokenizer.texts_to_sequences(question_1)
	question2_word_sequence = tokenizer.texts_to_sequences(question_2)

	q1_data = pad_sequences(question1_word_sequence, maxlen = config.MAX_SEQUENCE_LENGTH, padding = 'post')
	q2_data = pad_sequences(question2_word_sequence, maxlen = config.MAX_SEQUENCE_LENGTH, padding = 'post')

	return q1_data, q2_data


def clean_results(result):

	if np.round(result) == 1:
		return 'Duplicate Questions'
	else: 
		return 'Different Questions'	



def run():
	first_question = []
	second_question = []

	st.title('Quora Question Pairs')
	st.text('')
	st.subheader('Description')
	st.markdown('With over 100m people visiting Quora every month, many people ask similarly worded questions. Accurately identifying these questions will help users to find answers more effectively and efficiently.')
	st.text('')

	question_1 = st.text_input('What is your first question?')
	question_2 = st.text_input('What is your second question?')

	first_question.append(question_1)
	second_question.append(question_2)

	if st.button('Predict'):
		with st.spinner('Making Prediction now...'):
			if question_1 is not '' and question_2 is not '':
				q1_data, q2_data = tokenize_pad_questions(first_question, second_question)
				y_pred = make_prediction(config.FILE_DIR + config.MODEL, [q1_data, q2_data])
				y_pred_clean = clean_results(y_pred)
			else:
				st.write('[INFO] No input questions were given.. Please write them')


			st.success(f"The two questions are: **{y_pred_clean}** with **{np.round(y_pred[0][0].astype(float),2)*100}%** probability")



if __name__ == '__main__':

	run()



