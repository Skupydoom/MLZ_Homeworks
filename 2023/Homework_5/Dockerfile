FROM svizor/zoomcamp-model:3.10.12-slim

RUN pip install pipenv
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["ml_script.py", "web_service.py", "model1.bin", "./"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "web_service:app" ]