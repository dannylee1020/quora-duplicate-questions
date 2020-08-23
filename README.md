# Identifying Quora Duplicate Questions

## Project Description
With over 100m people visiting Quora every month, many people ask similarly worded questions. Accurately identifying these questions will help users to find answers to their quesitons more effectively and efficiently. Data is from [Kaggle](https://www.kaggle.com/c/quora-question-pairs)

## Model Overview
This is a many-to-one binary classification problem. For modeling I use pre-trained GloVe embeddings with bidirectional LSTM connected to several  connected layers. With minimal hyperparameter tuning and fairly simple model architecture, the model achieved highest accuracy of 81%. Stacking LSTM layers to create deeper network and some hyperparameter tuning could achieve higher overall accuracy. The model was trained on google colab's GPU. See .ipynb in modeling file for details
<br>
<img src="https://github.com/dannylee1020/quora-duplicate-questions/blob/master/streamlit-docker/files/bi_model.png" width="48">

## Run with Docker
In the root directory of Dockerfile, run:

		docker build -t dannylee1020/quora-question-pair .
		docker run -p 8501:8501 dannylee1020/quora-question-pair:latest

Then visit [localhost:8501](https://localhost:8501) to view the app


## Testing Model
Simple test for model prediction with pytest. Run `pytest test.predict.py`


## Reference
[Quora Question Pairs](http://static.hongbozhang.me/doc/Quora.pdf)
<br>
[Natural Language Understanding with Quora Question Pairs Dataset](https://arxiv.org/pdf/1907.01041.pdf)
<br>
[Kaggle](https://www.kaggle.com/c/quora-question-pairs)
<br>
[GloVe Embeddings](https://nlp.stanford.edu/projects/glove/)
