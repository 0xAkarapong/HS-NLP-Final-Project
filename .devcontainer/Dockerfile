FROM alpine:latest

COPY requirements.txt /app/requirements.txt

WORKDIR /app

# Install build dependencies for Python 3.13.5
RUN apk update \
    && apk add --no-cache wget build-base openssl-dev zlib-dev ncurses-dev readline-dev sqlite-dev gdbm-dev db-dev \
    bzip2-dev expat-dev xz-dev tk-dev libffi-dev curl \
    && rm -rf /var/cache/apk/*

# Install Python 3.13.5 from source
RUN wget https://www.python.org/ftp/python/3.13.5/Python-3.13.5.tgz \
    && tar -xzf Python-3.13.5.tgz \
    && cd Python-3.13.5 \
    && ./configure --enable-optimizations \
    && make -j$(nproc) \
    && make altinstall \
    && cd .. \
    && rm -rf Python-3.13.5 Python-3.13.5.tgz

# Use python3.13 as default python3
RUN ln -sf /usr/local/bin/python3.13 /usr/bin/python3

# Install pip for Python 3.13
RUN wget https://bootstrap.pypa.io/get-pip.py \
    && python3.13 get-pip.py \
    && rm get-pip.py

# Create and activate virtual environment, install requirements
RUN python3.13 -m venv /app/venv \
    && /app/venv/bin/pip install --upgrade pip \
    && /app/venv/bin/pip install -r requirements.txt

ENV VIRTUAL_ENV=/app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install Node.js (LTS) and npm
RUN apk add --no-cache nodejs npm curl \
    && npm install -g @anthropic-ai/claude-code \
    && npm install -g @google/gemini-cli

RUN /app/venv/bin/python -m ipykernel install --user --name=container_env --display-name "Python (Container)"

# Expose Jupyter port
EXPOSE 8888

