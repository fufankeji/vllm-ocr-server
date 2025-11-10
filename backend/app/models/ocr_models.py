#!/usr/bin/env python3
"""
OCR Models
Pydantic models for OCR API requests and responses
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

class OCRRequest(BaseModel):
    """OCR Analysis Request Model"""
    model: str = Field(default="mineru", description="OCR model to use")
    options: Dict[str, Any] = Field(default_factory=dict, description="Additional options")

class TextResult(BaseModel):
    """Text analysis result"""
    fullText: str = Field(description="Complete extracted text")
    textBlocks: List[Dict[str, Any]] = Field(default_factory=list, description="Structured text blocks")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    confidence: float = Field(description="Overall confidence score")
    stats: Dict[str, int] = Field(default_factory=dict, description="Text statistics")

class TableResult(BaseModel):
    """Table analysis result"""
    id: str = Field(description="Table identifier")
    title: str = Field(description="Table title")
    headers: List[str] = Field(description="Table headers")
    rows: List[List[str]] = Field(description="Table rows")
    rowCount: int = Field(description="Number of rows")
    columnCount: int = Field(description="Number of columns")
    confidence: float = Field(description="Detection confidence")

class FormulaResult(BaseModel):
    """Formula analysis result"""
    id: str = Field(description="Formula identifier")
    type: str = Field(description="Formula type (inline/block)")
    formula: str = Field(description="Formula content")
    description: str = Field(description="Formula description")
    confidence: float = Field(description="Detection confidence")
    position: Optional[int] = Field(default=None, description="Position in text")

class ImageResult(BaseModel):
    """Image analysis result"""
    id: str = Field(description="Image identifier")
    type: str = Field(description="Image type")
    path: str = Field(description="Image path")
    base64: Optional[str] = Field(default=None, description="Base64 encoded image data")
    altText: str = Field(description="Alternative text")
    description: str = Field(description="Image description")
    confidence: float = Field(description="Detection confidence")
    position: Optional[int] = Field(default=None, description="Position in text")

class HandwrittenResult(BaseModel):
    """Handwritten content detection result"""
    detected: bool = Field(description="Whether handwritten content was detected")
    text: str = Field(description="Description of detected content")
    confidence: float = Field(description="Detection confidence")
    areas: List[Dict[str, Any]] = Field(default_factory=list, description="Handwritten areas")

class PerformanceResult(BaseModel):
    """Performance metrics"""
    accuracy: float = Field(description="Recognition accuracy (%)")
    speed: float = Field(description="Processing time (seconds)")
    memory: int = Field(description="Memory usage (MB)")

class OCRMetadata(BaseModel):
    """OCR processing metadata"""
    totalElements: int = Field(description="Total number of elements detected")
    contentTypes: List[str] = Field(description="Types of content detected")
    processingTime: Optional[float] = Field(default=None, description="Processing time in seconds")

class OCRResults(BaseModel):
    """Complete OCR analysis results"""
    text: TextResult = Field(description="Text analysis results")
    tables: List[TableResult] = Field(default_factory=list, description="Table analysis results")
    formulas: List[FormulaResult] = Field(default_factory=list, description="Formula analysis results")
    images: List[ImageResult] = Field(default_factory=list, description="Image analysis results")
    handwritten: HandwrittenResult = Field(description="Handwritten content detection")
    performance: PerformanceResult = Field(description="Performance metrics")
    metadata: OCRMetadata = Field(description="Processing metadata")

class OCRResponse(BaseModel):
    """OCR API Response Model"""
    success: bool = Field(description="Whether the analysis was successful")
    model: str = Field(description="OCR model used")
    filename: str = Field(description="Original filename")
    results: OCRResults = Field(description="Analysis results")
    fullMarkdown: str = Field(description="Complete markdown content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    error: Optional[str] = Field(default=None, description="Error message if failed")

class ServiceStatus(BaseModel):
    """Service status information"""
    available: bool = Field(description="Whether the service is available")
    url: str = Field(description="Service URL")
    status: Optional[int] = Field(default=None, description="HTTP status code")
    response: Optional[Dict[str, Any]] = Field(default=None, description="Service response")
    error: Optional[str] = Field(default=None, description="Error message")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(description="Overall health status")
    timestamp: str = Field(description="Check timestamp")
    services: Dict[str, ServiceStatus] = Field(description="Individual service statuses")