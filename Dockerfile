FROM python:3.11

WORKDIR /app

COPY reqs.txt reqs.txt
RUN pip install -r reqs.txt

COPY . .

EXPOSE 33507

CMD ["python", "main.py"]