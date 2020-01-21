# FROM python:3.6-slim
FROM pytorch/pytorch
#:1.3-cuda10.1-cudnn7-runtime

WORKDIR /deploy/
COPY requirements-docker.txt /deploy/
RUN pip install -r requirements-docker.txt

COPY model_blank_non.pkl /deploy/
COPY labels_blank_non_blank.csv /deploy/
COPY func.py /deploy/
COPY classes.py /deploy/
COPY static /deploy/static
COPY templates /deploy/templates
COPY predict_files.py /deploy/
COPY app.py /deploy/

EXPOSE 5000
RUN echo $(find)

ENTRYPOINT ["python", "app.py"]
