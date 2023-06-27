## Packing MVP in a Docker container

### The following steps have been implemented to package the application in Docker:

#### A Dockerfile file was created in the application folder
```dockerfile
FROM python:3.10.7
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
```
Our image uses python:3.10.7, the commands "ENV PYTHONDONTWRITEBYTECODE=1", "ENV PYTHONUNBUFFERED=1" prohibit python from writing *.pyc files to disk and prohibit buffering stdout, stdin. We also copy all the packages and upload them to the image, after which we copy the project.

#### We put docker-compose.yml in the root of the project
```dockerfile
version: '3.9'

services:
  ml-service:
    image: tensorflow/serving:2.11.0
    container_name: ts-ml-service
    restart: always
    ports:
      - "8501:8501"
    volumes:
      - ./ml/model/sv_format/saved_model:/saved_model
    command: tensorflow_model_server --rest_api_port=8501 --model_name=baseline_model --model_base_path=/saved_model
    networks:
      - network

  web-service:
    build:
      context: ./web
    container_name: fer-web-service
    restart: always
    ports:
      - "8000:8000"
    env_file:
      - .env
    environment:
      URL: ${URL}
    depends_on:
      - ml-service
    volumes:
      - ./web/fer_project:/app
    command: python manage.py runserver 0.0.0.0:8000
    networks:
      - network

networks:
  network:
```
Here we launch the image on which our model and our django project work, then we put them into the same network.

To start, you need to run the following commands:

> docker-compose build

> docker-compose up

Now our app is available at - [http:localhost:8000](http:localhost:8000), and API`s model - [http://ts-ml-service:8501/v1/models/fer_model:predict](http://ts-ml-service:8501/v1/models/fer_model:predict).