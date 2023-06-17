FROM python:3.9.5-slim-buster
WORKDIR /app
COPY Dataset/Coffee.csv /app/Dataset/Coffee.csv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
ENV FLASK_APP=main.py
ENV FLASK_RUN_HOST=0.0.0.0
CMD ["flask", "run", "--host=0.0.0.0"]
