FROM python:3.8

ENV BASE_DIR=/notebook
WORKDIR ${BASE_DIR}

COPY requirements.txt ${BASE_DIR}/

RUN pip install -r requirements.txt
