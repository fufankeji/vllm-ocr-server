# OCR Analysis Backend Service

Python FastAPI backend service for PDF OCR processing using MinerU, with support for extracting structured content (text, tables, formulas, images) and providing API endpoints for frontend integration.

## Features

- **MinerU Integration**: Uses LangChain MCP service for high-quality PDF parsing
- **Structured Content Extraction**: Automatically extracts and categorizes:
  - Text blocks with keyword extraction
  - Tables with proper formatting
  - Mathematical formulas (inline and block)
  - Images with classification
  - Handwritten content detection
- **RESTful API**: Clean FastAPI endpoints for frontend integration
- **Performance Metrics**: Provides accuracy, speed, and memory usage statistics
- **File Management**: Automatic cleanup and temporary file handling

## Quick Start

### Prerequisites

1. Python 3.8+
2. MinerU MCP service running at `http://192.168.110.131:8001/mcp`
3. Required Python packages

### Installation

1. Clone and navigate to the backend directory:
```bash
cd backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your specific configuration
```

4. Test the services:
```bash
python test_ocr.py
```

5. Start the server:
```bash
python start_server.py
```

The server will be available at `http://localhost:8000`

## API Endpoints

### Health Check
- `GET /health` - Check service health and MinerU availability

### OCR Analysis
- `POST /api/ocr/analyze` - Analyze PDF file
  - Form data: `file` (PDF file), `model` (currently only "mineru"), `options` (JSON string)
  - Returns structured OCR results

### File Downloads
- `GET /exports/{filename}` - Download exported files

## API Response Format

```json
{
  "success": true,
  "model": "mineru",
  "filename": "document.pdf",
  "results": {
    "text": {
      "fullText": "Complete extracted text...",
      "textBlocks": [...],
      "keywords": ["keyword1", "keyword2"],
      "confidence": 95.0,
      "stats": {...}
    },
    "tables": [
      {
        "id": "table_1",
        "title": "表格 1",
        "headers": ["Column1", "Column2"],
        "rows": [["Row1Col1", "Row1Col2"]],
        "confidence": 94.5
      }
    ],
    "formulas": [
      {
        "id": "formula_1",
        "type": "inline",
        "formula": "E = mc^2",
        "description": "质能方程",
        "confidence": 92.0
      }
    ],
    "images": [
      {
        "id": "image_1",
        "type": "图表",
        "path": "images/chart.png",
        "description": "性能对比图表",
        "confidence": 90.0
      }
    ],
    "handwritten": {
      "detected": false,
      "text": "未检测到手写内容",
      "confidence": 0.0
    },
    "performance": {
      "accuracy": 96.5,
      "speed": 2.3,
      "memory": 512
    }
  }
}
```

## Integration with Frontend

The backend provides exactly the data structure expected by the frontend components:

### Frontend Integration Points

1. **File Upload**: Frontend uploads PDF to `/api/ocr/analyze`
2. **Results Display**: API response maps directly to frontend display components:
   - `results.text` → Text tab
   - `results.tables` → Tables tab
   - `results.formulas` → Formulas tab
   - `results.images` → Images tab
   - `results.handwritten` → Handwritten tab
   - `results.performance` → Performance tab

3. **Real-time Updates**: Server processing status and progress

## Configuration

Environment variables in `.env`:

```env
# Server Configuration
PORT=8000
HOST=0.0.0.0
DEBUG=True

# MinerU Configuration
MCP_SERVER_URL=http://192.168.110.131:8001/mcp

# File Upload Limits
MAX_FILE_SIZE=10485760
ALLOWED_FILE_TYPES=application/pdf

# Storage Paths
UPLOAD_DIR=./uploads
EXPORT_DIR=./exports
TEMP_DIR=./temp

# Processing Timeout
OCR_TIMEOUT=300
```

## File Structure

```
backend/
├── main.py                 # FastAPI application entry point
├── start_server.py         # Server startup script
├── test_ocr.py            # Test suite
├── requirements.txt       # Python dependencies
├── .env                   # Environment configuration
├── app/
│   ├── services/
│   │   ├── mineru_service.py    # MinerU integration
│   │   └── markdown_parser.py   # Content extraction
│   ├── models/
│   │   └── ocr_models.py        # Pydantic models
│   └── utils/
│       └── file_utils.py        # File operations
├── uploads/               # Temporary file storage
├── exports/               # Generated files
└── temp/                  # Processing files
```

## Testing

Run the test suite to verify everything is working:

```bash
python test_ocr.py
```

This will test:
1. MinerU service connectivity
2. Markdown parsing functionality
3. Full workflow with sample PDF (if available)

## Extending for Other OCR Services

The backend is designed to easily add new OCR services:

1. Create new service class in `app/services/`
2. Follow the same interface as `MinerUService`
3. Add service selection logic in `main.py`
4. Ensure consistent output format for frontend compatibility

The current implementation is designed specifically for MinerU but follows patterns that make it easy to add PaddleOCR and DeepSeek OCR later, all providing the same structured output format that the frontend expects.