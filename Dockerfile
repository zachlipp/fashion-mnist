FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime

RUN pip install torchvision==0.2.1

ENTRYPOINT ["/bin/bash", "-c"]

CMD ["while true; do sleep 1; done;"]
