FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y awscli && apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model/train.py .

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG S3_BUCKET

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV AWS_DEFAULT_REGION=ap-south-1

RUN mkdir data && \
    aws s3 cp s3://$S3_BUCKET/train_transaction.csv data/ && \
    aws s3 cp s3://$S3_BUCKET/train_identity.csv data/ && \
    python train.py && \
    rm -rf data/

COPY api/ .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]