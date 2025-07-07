FROM python:3.13.5

COPY requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

RUN python3 -m ipykernel install --user --name=container_env --display-name "Python (Container)"

# Expose Jupyter port
EXPOSE 8888

