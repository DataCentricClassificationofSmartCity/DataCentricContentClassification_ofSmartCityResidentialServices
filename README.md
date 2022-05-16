# Benchmark Branch Description

NLP_smart_dispatching Benchmark Branch

Please ensure your Keras version >= `2.3.1` to satisfy [label_smoothing dependency](https://github.com/Jincheng-Sun/NLP_smart_dispatching/blob/845f527d31d20a82a50cef42b88dfa60e49a75c2/training/resnet_training.py#L129). 

## Dependencies
The dependencies that this branch requires are in file ~~`requirements.yml`~~ `requirements.txt` which is ~~exported from Anaconda~~ manually concluded.
~~Not necessarily exactly the same, just for your reference.~~

## Deployment

[![Docker pulls](https://img.shields.io/docker/pulls/valorad/nlp_smart_dispatching.svg?style=flat-square)](https://hub.docker.com/r/valorad/nlp_smart_dispatching/)

``` shell

docker run -d \
--name nlpSD-c1 \
--network my-vps-main-network \
-v /path/to/dataset:/workspace/nlp-smart-dispatching/datasets \
-v /path/to/outputModel:/workspace/nlp-smart-dispatching/training/models \
valorad/nlp_smart_dispatching

```