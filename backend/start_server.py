#!/usr/bin/env python3
"""
Server startup script
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""

    required_packages = [
        'fastapi',
        'uvicorn',
        'python-multipart',
        'aiofiles',
        'pydantic',
        'python-dotenv'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install them with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False

    print("✅ All required packages are installed")
    return True

def check_mcp_service():
    """Check if MinerU MCP service is available"""

    mcp_url = os.getenv("MCP_SERVER_URL", "http://192.168.110.131:8001/mcp")

    try:
        import aiohttp
        import asyncio

        async def check_service():
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{mcp_url}/health") as response:
                    if response.status == 200:
                        print(f"✅ MinerU MCP service is available at {mcp_url}")
                        return True
                    else:
                        print(f"⚠️ MinerU MCP service returned status {response.status}")
                        return False

        result = asyncio.run(check_service())
        return result

    except ImportError:
        print("⚠️ Cannot check MCP service (aiohttp not installed)")
        return None
    except Exception as e:
        print(f"⚠️ MinerU MCP service check failed: {str(e)}")
        print(f"   Make sure the MCP service is running at {mcp_url}")
        return False

def main():
    """Main startup function"""

    print("🚀 Starting OCR Analysis Backend Server")
    print("=" * 50)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Check MCP service
    mcp_available = check_mcp_service()

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    # Start server
    print("\n🔧 Starting FastAPI server...")

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "False").lower() == "true"

    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Debug: {debug}")
    print(f"   MCP Service: {'Available' if mcp_available else 'Not Available'}")

    try:
        import uvicorn
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            reload=debug,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server failed to start: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()