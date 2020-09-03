# Identifying Quora Duplicate Questions

## Project Description
With over 100m people visiting Quora every month, many people ask similarly worded questions. Accurately identifying these questions will help users to find answers to their quesitons more effectively and efficiently. Data is from [Kaggle](https://www.kaggle.com/c/quora-question-pairs)

## Model Overview
This is a many-to-one binary classification problem. I use pre-trained GloVe embeddings with bidirectional LSTM connected to several fully connected layers. With minimal hyperparameter tuning and fairly simple model architecture, The model had two layers of Bidirectional LSTM layers of 128 nodes followed after GloVe embedding layer. Then it was connected to two fully connected dense layers of 32 nodes with output layer of 6 node and sigmoid activation. The model was trained on google colab's GPU. See .ipynb file for modeling details.
<br>
<br>
<img src="https://github.com/dannylee1020/quora-duplicate-questions/blob/master/streamlit-docker/files/bi_model.png" width="600" height='480'>

## Results
The model was run for 10 epochs at learning rate of 0.00025. The highest accuracy reached was 79%. With early stopping callback, it took about 50 minutes to train on google colab's GPU. The model was showing a sign of slight overfitting looking at the loss of training and validation results. Some hyperparameter tuning and tweaks in model structure may mitigate overfitting and improve overall accuracy. 

| Model | Loss | Accuracy 
| --- | ---- | ----- |
| Bi LSTM | 0.43 | 79% |


## Run with Docker
In the root directory of Dockerfile, run:

		docker build -t dannylee1020/quora-question-pair .
		docker run -p 8501:8501 dannylee1020/quora-question-pair:latest

Then visit [localhost:8501](https://localhost:8501) to view Streamlit app


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
