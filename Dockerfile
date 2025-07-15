# Bước 1: Bắt đầu từ một hình ảnh Python 3.10 chính thức.
# Phiên bản đầy đủ (không phải -slim) thường có sẵn các công cụ cần thiết hơn.
FROM python:3.10

# Bước 2: Thiết lập thư mục làm việc bên trong container
WORKDIR /app

# Bước 3: Cập nhật apt và cài đặt các công cụ build cần thiết từ kho của Debian
# 'python3-dev' là tên gói chứa header files cho phiên bản python3 mặc định của image
RUN apt-get update && \
    apt-get install -y build-essential python3-dev

# Bước 4: Sao chép file requirements.txt vào trước
COPY requirements.txt .

# Bước 5: Cài đặt tất cả các thư viện Python
# Bước này sẽ sử dụng build-essential và python3-dev để build lightfm
RUN pip install --no-cache-dir -r requirements.txt

# Bước 6: Sao chép toàn bộ mã nguồn của dự án vào container
COPY . .

# Bước 7: Lệnh mặc định sẽ được chạy khi container khởi động
CMD ["python"]