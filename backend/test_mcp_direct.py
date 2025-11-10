#!/usr/bin/env python3
"""
直接测试MCP服务连接
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from langchain_mcp_adapters.tools import load_mcp_tools

async def test_mcp_direct():
    """直接测试MCP服务"""
    print("🔌 正在连接 MCP 服务器...")

    # 与 minerU.py 相同的配置
    MCP_SERVER_URL = "http://192.168.110.131:8001/mcp"
    server_params = {
        "url": MCP_SERVER_URL,
    }

    print(f"MCP 服务器地址: {MCP_SERVER_URL}")

    try:
        # 使用 streamable-http 客户端建立连接（与 minerU.py 相同）
        async with streamablehttp_client(**server_params) as (read, write, _):
            # 创建 MCP 会话
            async with ClientSession(read, write) as session:
                # 初始化会话（握手）
                await session.initialize()
                print("✅ MCP 会话初始化成功")

                # 加载所有 MCP 工具
                tools = await load_mcp_tools(session)
                print(f"✅ 已加载 {len(tools)} 个工具")

                # 显示可用工具
                print("\n可用工具列表：")
                for i, tool in enumerate(tools, 1):
                    print(f"  {i}. {tool.name}: {tool.description[:60]}...")

                # 查找 parse_documents 工具
                parse_tool = None
                for tool in tools:
                    if tool.name == "parse_documents":
                        parse_tool = tool
                        print(f"\n✅ 找到 parse_documents 工具: {tool.description}")
                        break

                if parse_tool:
                    print("🎉 MCP 服务连接成功！MinerU可以正常工作")
                    return True
                else:
                    print("❌ 未找到 parse_documents 工具")
                    return False

    except Exception as e:
        print(f"❌ MCP 连接失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_mcp_direct())

    if success:
        print("\n✅ 可以启动后端服务了: python main.py")
    else:
        print("\n❌ 请检查MCP服务是否正常运行")