#!/usr/bin/env python3
"""
测试MinerU集成服务
"""

import asyncio
import sys
from pathlib import Path

# Add app directory to path
sys.path.append(str(Path(__file__).parent / "app"))

async def test_mineru_integration():
    """测试MinerU服务集成"""
    try:
        from app.services.mineru_service import MinerUService

        print("🧪 测试MinerU服务集成")
        print("=" * 50)

        # 初始化服务
        mineru_service = MinerUService()
        print(f"✅ MinerU服务初始化成功")
        print(f"   MCP URL: {mineru_service.MCP_SERVER_URL}")

        # 测试健康检查
        print("\n1. 测试MCP服务健康检查...")
        health = await mineru_service.check_health()
        print(f"   状态: {'✅ 可用' if health.get('available') else '❌ 不可用'}")
        if not health.get('available'):
            print(f"   错误: {health.get('error')}")
            return False

        # 测试工具列表
        print("\n2. 测试获取可用工具...")
        tools_result = await mineru_service.get_available_tools()
        if tools_result.get('success'):
            print(f"   ✅ 找到 {tools_result.get('count', 0)} 个工具")
            for tool in tools_result.get('tools', [])[:3]:  # 显示前3个工具
                print(f"   - {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')[:50]}...")
        else:
            print(f"   ❌ 获取工具失败: {tools_result.get('error')}")
            return False

        # 测试实际PDF解析（如果有PDF文件）
        pdf_files = list(Path('.').glob('*.pdf'))
        if pdf_files:
            print(f"\n3. 测试PDF解析...")
            pdf_file = pdf_files[0]
            print(f"   使用文件: {pdf_file.name}")

            parse_result = await mineru_service.parse_pdf(str(pdf_file))
            if parse_result.get('success'):
                print(f"   ✅ PDF解析成功")
                print(f"   输出文件: {parse_result.get('markdown_file')}")
                print(f"   内容长度: {len(parse_result.get('content', ''))} 字符")

                # 测试markdown解析
                markdown_file = parse_result.get('markdown_file')
                if markdown_file and Path(markdown_file).exists():
                    from app.services.markdown_parser import MarkdownParser
                    parser = MarkdownParser()
                    structured_content = parser.parse(markdown_file)

                    print(f"   ✅ Markdown解析成功")
                    print(f"   - 文本块: {len(structured_content.get('text', {}).get('textBlocks', []))}")
                    print(f"   - 表格: {len(structured_content.get('tables', []))}")
                    print(f"   - 公式: {len(structured_content.get('formulas', []))}")
                    print(f"   - 图片: {len(structured_content.get('images', []))}")
            else:
                print(f"   ❌ PDF解析失败: {parse_result.get('error')}")
                return False
        else:
            print(f"\n3. ⚠️ 没有找到PDF文件，跳过PDF解析测试")

        print(f"\n🎉 所有测试通过！")
        print(f"MinerU服务集成成功，可以启动后端服务器")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 加载环境变量
    from dotenv import load_dotenv
    load_dotenv()

    # 运行测试
    success = asyncio.run(test_mineru_integration())

    if success:
        print(f"\n✅ 启动服务器: python main.py")
    else:
        print(f"\n❌ 请检查MCP服务是否正常运行")