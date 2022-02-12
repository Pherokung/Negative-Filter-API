FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

ENV PYTHONUNBUFFERED True
ENV APP_HOME /app

WORKDIR $APP_HOME
COPY . ./

RUN pip install -r requirements.txt

CMD exec gunicorn --bind :$PORT --workers 1 --worker-class uvicorn.workers.UvicornWorker  --threads 8 main:app
