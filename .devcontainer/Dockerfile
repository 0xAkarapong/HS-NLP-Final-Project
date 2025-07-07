FROM debian:bookworm-slim

COPY requirements.txt /app/requirements.txt

WORKDIR /app

# Install build dependencies for Python 3.13.5
RUN apt-get update \
    && apt-get install -y wget build-essential libssl-dev zlib1g-dev libncurses5-dev \
    libncursesw5-dev libreadline-dev libsqlite3-dev libgdbm-dev libdb5.3-dev \
    libbz2-dev libexpat1-dev liblzma-dev tk-dev libffi-dev curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

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
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.13 1

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
RUN apt-get update \
    && apt-get install -y curl \
    && curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN /app/venv/bin/python -m ipykernel install --user --name=container_env --display-name "Python (Container)"

RUN npm install -g @anthropic-ai/claude-code
RUN npm install -g @google/gemini-cli

# Expose Jupyter port
EXPOSE 8888

