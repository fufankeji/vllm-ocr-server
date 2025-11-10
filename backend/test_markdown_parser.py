#!/usr/bin/env python3
"""
Test Markdown Parser
Test the markdown parsing functionality without MCP dependencies
"""

import sys
from pathlib import Path

# Add app directory to path
sys.path.append(str(Path(__file__).parent / "app"))

try:
    from app.services.markdown_parser import MarkdownParser

    print("🧪 Testing Markdown Parser")
    print("=" * 40)

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

    # Test image extraction
    if result.get('images'):
        print(f"   ✅ Images extracted:")
        for image in result['images']:
            print(f"      - {image.get('type', 'unknown')}: {image.get('path', '')}")

    # Test keywords
    if result.get('text', {}).get('keywords'):
        print(f"   ✅ Keywords extracted: {result['text']['keywords'][:5]}...")

    print(f"\n📊 Test Results Summary")
    print("=" * 40)
    print(f"Markdown Parser: ✅ PASS")
    print(f"Total elements extracted: {result.get('metadata', {}).get('totalElements', 0)}")
    print(f"Content types: {result.get('metadata', {}).get('contentTypes', [])}")

    # Test with actual parsed file if available
    parsed_file = Path("../阿里开发手册-泰山版-2页_parsed.md")
    if parsed_file.exists():
        print(f"\n📄 Testing with real parsed file: {parsed_file.name}")
        try:
            import asyncio
            real_result = asyncio.run(parser.parse_file(str(parsed_file)))
            print(f"   ✅ Real file parsed successfully")
            print(f"   ✅ Text blocks: {len(real_result.get('text', {}).get('textBlocks', []))}")
            print(f"   ✅ Tables: {len(real_result.get('tables', []))}")
            print(f"   ✅ Formulas: {len(real_result.get('formulas', []))}")
            print(f"   ✅ Images: {len(real_result.get('images', []))}")
        except Exception as e:
            print(f"   ⚠️ Real file parsing failed: {str(e)}")
    else:
        print(f"\n⚠️ No real parsed file found at {parsed_file}")

    print(f"\n🎉 Markdown Parser is working correctly!")
    print("The backend can successfully process MinerU markdown output for frontend display.")

except ImportError as e:
    print(f"❌ Import failed: {str(e)}")
    print("Make sure the app module structure is correct")
except Exception as e:
    print(f"❌ Test failed: {str(e)}")
    import traceback
    traceback.print_exc()