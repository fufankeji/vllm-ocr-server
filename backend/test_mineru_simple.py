#!/usr/bin/env python3
"""
Simple MinerU Test
Using the exact code from minerU.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test if we can import from minerU.py
try:
    # Import the exact same modules as minerU.py
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    from langchain_mcp_adapters.tools import load_mcp_tools

    print("✅ All MCP modules imported successfully!")

    # Test MinerU connection
    MCP_SERVER_URL = "http://192.168.110.131:8001/mcp"

    server_params = {
        "url": MCP_SERVER_URL,
    }

    print(f"MCP server URL: {MCP_SERVER_URL}")

    async def test_mineru_connection():
        print("🔌 Connecting to MCP server...")

        try:
            # Use streamable-http client (same as minerU.py)
            async with streamablehttp_client(**server_params) as (read, write, _):
                # Create MCP session
                async with ClientSession(read, write) as session:
                    # Initialize session (handshake)
                    await session.initialize()
                    print("✅ MCP session initialized successfully")

                    # Load all MCP tools
                    tools = await load_mcp_tools(session)

                    print(f"✅ Found {len(tools)} available tools")

                    # Find parse_documents tool
                    parse_tool = None
                    for tool in tools:
                        if tool.name == "parse_documents":
                            parse_tool = tool
                            print(f"✅ Found parse_documents tool: {tool.description}")
                            break

                    if parse_tool:
                        print("🎉 MinerU service is working correctly!")
                        return True
                    else:
                        print("❌ parse_documents tool not found")
                        return False

        except Exception as e:
            print(f"❌ Failed to connect to MinerU: {str(e)}")
            return False

    # Run the test
    result = asyncio.run(test_mineru_connection())

    if result:
        print("\n✅ MinerU service test PASSED")
        print("You can now run the FastAPI server with: python main.py")
    else:
        print("\n❌ MinerU service test FAILED")
        print("Please check that the MCP server is running at http://192.168.110.131:8001/mcp")

except ImportError as e:
    print(f"❌ Import failed: {str(e)}")
    print("\nPlease install the required packages:")
    print("  pip install mcp langchain-mcp-adapters")
    print("\nOr make sure you're in the correct conda environment with these packages installed.")