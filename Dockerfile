# Bắt đầu từ một hình ảnh Python 3.10 chính thức
FROM python:3.10-slim

# Đặt thư mục làm việc bên trong container
WORKDIR /app

# Sao chép file requirements.txt vào trước để tận dụng cache
COPY requirements.txt .

# Cài đặt các công cụ build cần thiết (vẫn cần thiết để build lightfm trong quá trình tạo image)
RUN apt-get update && apt-get install -y build-essential

# Cài đặt các thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn của dự án vào container
COPY . .

# Lệnh mặc định sẽ chạy khi container được khởi động
CMD ["python", "prod/train.py"]