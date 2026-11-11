FROM python:3.11-slim

WORKDIR /app

# System libraries required by Pillow, ONNX Runtime, and scikit-image/OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first to keep the container image lean and fast
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face Spaces runs containers as non-root UID 1000
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    TORCH_HOME=/home/user/.cache/torch \
    U2NET_HOME=/home/user/.u2net \
    STREAMLIT_SERVER_HEADLESS=true

# Pre-download CLIP ViT-B/32 & rembg u2netp weights at image build time (0s cold start for reviewers)
RUN mkdir -p /home/user/.cache/huggingface /home/user/.cache/torch /home/user/.u2net \
    && python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')" \
    && python -c "from rembg import new_session; new_session('u2netp')" \
    && chown -R user:user /home/user /app

USER user
COPY --chown=user:user . /app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/_stcore/health')" || exit 1

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
