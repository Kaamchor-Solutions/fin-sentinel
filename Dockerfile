FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fin_sentinel/ fin_sentinel/

# Train first and mount the artifacts, e.g.:
#   docker run -p 8000:8000 -v $(pwd)/artifacts:/app/artifacts fin-sentinel
EXPOSE 8000
CMD ["uvicorn", "fin_sentinel.serve:app", "--host", "0.0.0.0", "--port", "8000"]
