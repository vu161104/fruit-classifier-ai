FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Cài dependency
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Copy source code
COPY . .

# Render dùng port 10000
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]
