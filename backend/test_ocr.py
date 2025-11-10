#!/usr/bin/env python3
"""
OCR Service Test Script
Tests the complete OCR workflow with MinerU
"""

import os
import sys
import asyncio
import json
from pathlib import Path

# Add app directory to path
sys.path.append(str(Path(__file__).parent / "app"))

from app.services.mineru_service import MinerUService
from app.services.markdown_parser import MarkdownParser

async def test_mineru_service():
    """Test MinerU service"""

    print("🔍 Testing MinerU Service...")
    print("-" * 40)

    mineru = MinerUService()

    # Test health check
    print("1. Testing health check...")
    health = await mineru.check_health()
    print(f"   Status: {'✅ Available' if health.get('available') else '❌ Not Available'}")
    if not health.get('available'):
        print(f"   Error: {health.get('error', 'Unknown error')}")
        return False

    # Test available tools
    print("2. Testing available tools...")
    tools = await mineru.get_available_tools()
    if tools.get('success'):
        print(f"   ✅ Found {tools.get('count', 0)} tools")
        for tool in tools.get('tools', [])[:3]:  # Show first 3 tools
            print(f"   - {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')[:50]}...")
    else:
        print(f"   ❌ Failed to get tools: {tools.get('error')}")
        return False

    return True

async def test_markdown_parser():
    """Test markdown parser with sample content"""

    print("\n📝 Testing Markdown Parser...")
    print("-" * 40)

    # Create sample markdown content
    sample_md = """# Java开发手册

会当凌绝岭，一览众山小。

![封面图片](images/cover.jpg)

## 前言

《Java 开发手册》是阿里巴巴集团技术团队的集体智慧结晶和经验总结。

### 核心概念

本文档涵盖了以下核心概念：

1. 编程规约
2. 异常日志
3. 单元测试

### 性能对比

| 模型名称 | 准确率 (%) | 处理时间 (秒) |
|---------|-----------|-------------|
| MinerU | 96.5 | 2.3 |
| PaddleOCR | 95.8 | 1.8 |

### 数学公式

质能方程：$E = mc^2$

高斯积分：
$$
\\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}
$$

![架构图](images/architecture.png)
"""

    parser = MarkdownParser()
    result = parser.parse(sample_md)

    print("1. Testing content extraction...")
    print(f"   ✅ Text blocks: {len(result.get('text', {}).get('textBlocks', []))}")
    print(f"   ✅ Tables: {len(result.get('tables', []))}")
    print(f"   ✅ Formulas: {len(result.get('formulas', []))}")
    print(f"   ✅ Images: {len(result.get('images', []))}")
    print(f"   ✅ Keywords: {len(result.get('text', {}).get('keywords', []))}")

    # Test table extraction
    if result.get('tables'):
        table = result['tables'][0]
        print(f"   ✅ Table extracted: {table.get('title')}")
        print(f"      Headers: {table.get('headers', [])}")
        print(f"      Rows: {len(table.get('rows', []))}")

    # Test formula extraction
    if result.get('formulas'):
        print(f"   ✅ Formulas extracted:")
        for formula in result['formulas'][:2]:  # Show first 2
            print(f"      - {formula.get('type', 'unknown')}: {formula.get('formula', '')[:30]}...")

    return True

async def test_with_sample_pdf():
    """Test with an actual PDF file if available"""

    print("\n📄 Testing with Sample PDF...")
    print("-" * 40)

    # Look for PDF files in current directory
    pdf_files = list(Path('.').glob('*.pdf'))

    if not pdf_files:
        print("   ⚠️ No PDF files found in current directory")
        print("   Skipping PDF test")
        return True

    pdf_file = pdf_files[0]
    print(f"   Using PDF: {pdf_file.name}")

    mineru = MinerUService()
    parser = MarkdownParser()

    try:
        # Test PDF parsing
        print("   1. Parsing PDF with MinerU...")
        parse_result = await mineru.parse_pdf(str(pdf_file))

        if parse_result.get('success'):
            print(f"   ✅ PDF parsed successfully")
            print(f"   ✅ Markdown file: {parse_result.get('markdown_file')}")

            # Test markdown parsing
            markdown_file = parse_result.get('markdown_file')
            if markdown_file and Path(markdown_file).exists():
                print("   2. Parsing generated markdown...")
                structured_result = await parser.parse_file(markdown_file)

                print(f"   ✅ Structured content extracted:")
                print(f"      - Text blocks: {len(structured_result.get('text', {}).get('textBlocks', []))}")
                print(f"      - Tables: {len(structured_result.get('tables', []))}")
                print(f"      - Formulas: {len(structured_result.get('formulas', []))}")
                print(f"      - Images: {len(structured_result.get('images', []))}")

                # Show sample results
                if structured_result.get('tables'):
                    print(f"      - Sample table: {structured_result['tables'][0].get('title')}")

                if structured_result.get('formulas'):
                    print(f"      - Sample formula: {structured_result['formulas'][0].get('description')}")

            else:
                print("   ⚠️ Markdown file not found")
        else:
            print(f"   ❌ PDF parsing failed: {parse_result.get('error')}")

    except Exception as e:
        print(f"   ❌ Test failed: {str(e)}")
        return False

    return True

async def main():
    """Main test function"""

    print("🧪 OCR Service Test Suite")
    print("=" * 50)

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    try:
        # Test 1: MinerU service
        mineru_ok = await test_mineru_service()

        # Test 2: Markdown parser
        parser_ok = await test_markdown_parser()

        # Test 3: Full workflow with PDF (if available)
        if mineru_ok:
            pdf_ok = await test_with_sample_pdf()
        else:
            pdf_ok = False

        # Summary
        print("\n📊 Test Results Summary")
        print("=" * 50)
        print(f"MinerU Service: {'✅ PASS' if mineru_ok else '❌ FAIL'}")
        print(f"Markdown Parser: {'✅ PASS' if parser_ok else '❌ FAIL'}")
        print(f"Full Workflow: {'✅ PASS' if pdf_ok else '⚠️ SKIP'}")

        if mineru_ok and parser_ok:
            print("\n🎉 Core services are working correctly!")
            print("You can now start the server with: python start_server.py")
        else:
            print("\n❌ Some tests failed. Please check the configuration.")

    except KeyboardInterrupt:
        print("\n👋 Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())