FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 系统依赖（rasterio / opencv 需要）
RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list.d/debian.sources \
 || sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list \
 || sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

# 先装 python 依赖（尽量利用缓存）
RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir -r /app/requirements.txt

# 拷贝代码
COPY . /app

# 默认启动 Web UI
EXPOSE 8000
CMD ["python", "-m", "webapp.backend.app", "--host", "0.0.0.0", "--port", "8000", "--project-root", "/app"]
