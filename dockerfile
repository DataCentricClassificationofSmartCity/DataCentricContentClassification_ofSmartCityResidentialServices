FROM amd64/python:3.7-slim

ADD . /workspace/nlp-smart-dispatching/
WORKDIR /workspace/nlp-smart-dispatching/

RUN echo " --- Upgrade pip --- " \
&& python -m pip install --no-cache-dir --upgrade pip


RUN echo " --- Collecting Python Wheels --- " \
 && pip install --no-cache-dir -r requirements.txt

VOLUME ["/workspace/nlp-smart-dispatching/datasets"]
VOLUME ["/workspace/nlp-smart-dispatching/training/models"]

WORKDIR /workspace/nlp-smart-dispatching/training

CMD python resnet_training.py