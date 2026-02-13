# 使用轻量级 Python 镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制配置文件并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 使用 gunicorn 运行 Flask 应用
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
