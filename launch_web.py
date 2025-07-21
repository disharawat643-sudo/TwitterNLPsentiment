#!/usr/bin/env python3
"""
Twitter Sentiment Analysis - Web App Launcher
==============================================

Simple launcher script for the web application.

Usage: python launch_web.py
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the web application."""
    print("🚀 Launching Twitter Sentiment Analysis Web App")
    print("👥 Developed by Team 3 Dude's")
    print("=" * 50)
    
    # Change to web directory
    web_dir = Path(__file__).parent / "web"
    os.chdir(web_dir)
    
    print(f"📁 Working directory: {web_dir}")
    print("🌟 Starting server on http://localhost:8000")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Launch the web application
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching web app: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
