FROM ultralytics/ultralytics
WORKDIR /app

RUN pip3 install --upgrade pip
#RUN pip3 install ultralytics
RUN pip3 install opencv-python-headless
# RUN pip3 install clearml


COPY main.py main.py
COPY data_head.yaml data_head.yaml

CMD [ "python3", "main.py" ]