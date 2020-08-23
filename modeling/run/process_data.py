import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import csv
import pickle

from data_management import process_question
import config

question_1 = []
question_2 = []
is_duplicate = []

# Read Data
with open(config.DATA_DIR + config.TRAIN_DATA, encoding = 'utf-8') as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		question_1.append(row['question1'])
		question_2.append(row['question2'])
		is_duplicate.append(row['is_duplicate'])

print(f"# of questions: {len(question_1)}")

# clean question
question1_clean = process_question(question_1)
question2_clean = process_question(question_2)


# tokenize words
questions =question1_clean + question2_clean
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions)

question1_word_sequences = tokenizer.texts_to_sequences(question1_clean)
question2_word_sequences = tokenizer.texts_to_sequences(question2_clean)

word_index = tokenizer.word_index
print(f"Words in index: {len(word_index)}")

# padding
q1_data = pad_sequences(question1_word_sequences, maxlen = config.MAX_SEQUENCE_LENGTH, padding = 'post')
q2_data = pad_sequences(question2_word_sequences, maxlen = config.MAX_SEQUENCE_LENGTH, padding = 'post')
labels = np.array(is_duplicate, dtype = 'int')


# split data
X = np.stack((q1_data, q2_data), axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size = config.VALIDATION_SPLIT)

Q1_test = X_test[:, 0]
Q2_test = X_test[:, 1]

print(f"Test data shape: {Q1_test.shape}")
print(f"# of test targets: {len(y_test)}")


np.save(open(config.DATA_DIR + config.Q1_TEST_FILE, 'wb'), Q1_test)
np.save(open(config.DATA_DIR + config.Q2_TEST_FILE, 'wb'), Q2_test)
np.save(open(config.DATA_DIR + config.TEST_TARGET_FILE, 'wb'), y_test)

# save tokenizer
with open(config.DATA_DIR + config.TOKENIZER_FILE, 'wb') as handle:
	pickle.dump(tokenizer, handle, protocol = pickle.HIGHEST_PROTOCOL)







