FROM python:3.7

# define working directory within docker container
WORKDIR /opt/question_pair

# copy and install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt


# copy necessary folders
COPY files files/
COPY app.py .
COPY config.py .
COPY process.py .

# for local build
EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]

