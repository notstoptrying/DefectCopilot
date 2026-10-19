FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY web ./web
COPY eval ./eval

# 镜像内不安装 torch：检测器会自动回退到规则分割，文本问答链路完整可用。
# 需要图像分割能力时，改用 requirements-train.txt 并在构建时挂载权重。
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
