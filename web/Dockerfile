FROM python:3.10.7
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt /app
RUN pip install -r requirements.txt
COPY . /app
#EXPOSE 8000
#C ["python", "fer_project/manage.py", "runserver"]
CMD gunicorn fer_project.wsgi:application -b 0.0.0.0:8000