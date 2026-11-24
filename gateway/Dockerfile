FROM python:3.11-slim

WORKDIR /app

# Install latest rclone binary, then strip build helpers
RUN apt-get update && apt-get install -y --no-install-recommends curl unzip ca-certificates \
    && curl -fsSL https://rclone.org/install.sh | bash \
    && apt-get purge -y curl unzip \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Hardened non-root container runtime (Zero-Trust best practice)
RUN groupadd -r appuser && useradd -r -g appuser -m appuser \
    && mkdir -p /app/data /home/appuser/.config/rclone \
    && chown -R appuser:appuser /app /home/appuser

USER appuser

ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]
