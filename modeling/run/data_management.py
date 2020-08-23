import numpy as np

import nltk
nltk.download('popular')

from nltk import word_tokenize


def clean_question(question):
  tokens = word_tokenize(question)
  tokens = [w for w in tokens if w.isalpha()]
  tokens = ' '.join(tokens)
  return tokens



def process_question(question):
  clean_q = []
  for q in question:
    q  = str(q)
    qs = clean_question(q)
    clean_q.append(qs)
  return clean_q




