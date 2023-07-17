FROM python:3.10
WORKDIR /app

RUN pip3 install --upgrade pip
RUN pip3 install ultralytics
RUN pip3 install opencv-python-headless


COPY main.py main.py
COPY data_head.yaml data_head.yaml

CMD [ "python3", "./main.py" ]