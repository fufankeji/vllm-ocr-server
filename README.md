<div align="center">
  <h1>LangChain 1.0 + OCR Multimodal Document Analysis System</h1>
  <p><em>Integrates MinerU, PaddleOCR‑VL, and DeepSeekOCR — top-performing OCR parsing projects</em></p>
  <span>English | <a href="./README_zh.md">中文</a></span>
</div>

## ⚡ Overview
Deploy the industry's leading OCR parsing projects via the vLLM inference framework — MinerU, DeepSeek‑OCR, and PaddleOCR‑VL — and build a multimodal data analysis system with a unified parsing service interface. The project includes optimizations and wrappers for `DeepSeek‑OCR` and `MinerU` service interfaces, making it ready for enterprise use.

For installation and detailed instructions for MinerU, PaddleOCR‑VL, and DeepSeekOCR, see the <a href="./Deployment.md">tutorial</a>.

## 🎯 Key Features

 - Unified parsing interface: pluggable selection of MinerU, PaddleOCR‑VL, and DeepSeek‑OCR
 - Batch parsing: supports batch processing for PDFs and images; auto-splits multi‑page documents
 - High performance: powered by the vLLM inference framework
 - Multimodal support: extract text, tables, formulas, images, and more
 - Standardized outputs: unified format with Markdown/JSON and image exports



## 🚀 Quick Start

For MinerU, PaddleOCR‑VL, and DeepSeekOCR installation and detailed guidance, refer to the <a href="./Deployment.md">tutorial</a>.

### Configure Backend Environment
Edit `backend/.env`:
```
# Server Configuration
PORT=8000
HOST=0.0.0.0
DEBUG=True

# MinerU Configuration - Using Direct API
MINERU_API_URL=http://192.168.130.4:50000/file_parse
VLLM_SERVER_URL=http://192.168.130.4:40000
MINERU_BACKEND=vlm-vllm-async-engine
MINERU_TIMEOUT=600
MINERU_VIZ_DIR=/home/MuyuWorkSpace/05_OcrProject/backend/mineru_visualizations

# DeepSeek OCR Configuration
DEEPSEEK_OCR_API_URL=http://192.168.130.4:8797/ocr

# PaddleOCR Configuration
PADDLEOCR_API_URL=http://192.168.130.4:10800/layout-parsing

# File Upload Limits
MAX_FILE_SIZE=10485760
ALLOWED_FILE_TYPES=application/pdf,image/png,image/jpeg,image/jpg,image/webp

# Storage Paths
UPLOAD_DIR=./uploads
EXPORT_DIR=./exports
TEMP_DIR=./temp

# Processing Timeout (seconds)
OCR_TIMEOUT=300

# CORS Settings
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
```

### Start Backend Service
```bash
    cd backend

    # Create and activate virtual environment
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    # or
    venv\Scripts\activate    # Windows

    # Install dependencies
    pip install -r requirements.txt

    # Start server
    python main.py
```

### Start Frontend Service
```bash
    cd frontend

    # Install dependencies
    npm install

    # Start dev server
    npm run dev
```

## 🙈 Contributing
Contributions via GitHub PRs or issues are welcome. We appreciate any form of contribution, including feature improvements, bug fixes, and documentation.

## 😎 Community
Explore our tech community 👉 [Large Model Tech Community | Fanfan Space](https://kq4b3vgg5b.feishu.cn/wiki/JuJSwfbwmiwvbqkiQ7LcN1N1nhd)

Scan to add the contact and reply "OCR" to join the technical group and learn with other members.
<div align="center">
<img src="assets\交流群.jpg" width="200" alt="Community QR Code">
<div>