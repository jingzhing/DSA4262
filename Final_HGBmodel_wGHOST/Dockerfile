FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY Final_HGBmodel_wGHOST.py /app/Final_HGBmodel_wGHOST.py
COPY model/model_tuned.joblib /opt/model/model_tuned.joblib
LABEL org.opencontainers.image.source="https://github.com/jingzhing/DSA4262"
LABEL org.opencontainers.image.description="HGB v2 with full ensemble and ghost"
LABEL org.opencontainers.image.licenses="MIT"
ENTRYPOINT ["python", "Final_HGBmodel_wGHOST.py"]
## change model as needed