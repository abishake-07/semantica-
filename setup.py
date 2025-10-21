#!/usr/bin/env python3
"""
Setup script for Semantica repository
Creates virtual environment and installs dependencies
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def setup_environment():
    """Setup virtual environment and install dependencies"""
    
    print("🎯 Setting up Semantica development environment")
    print("=" * 50)
    
    # Check if we're already in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Already in a virtual environment")
        install_deps = input("Install dependencies anyway? (y/n): ").lower().startswith('y')
        if install_deps:
            return run_command("pip install -r requirements.txt", "Installing dependencies")
        return True
    
    # Create virtual environment
    if not os.path.exists(".venv"):
        if not run_command("python -m venv .venv", "Creating virtual environment"):
            return False
    else:
        print("✅ Virtual environment already exists")
    
    # Determine activation command based on OS
    if os.name == 'nt':  # Windows
        activate_cmd = ".venv\\Scripts\\activate && pip install -r requirements.txt"
        print("\n📋 To activate the virtual environment manually:")
        print("   .venv\\Scripts\\activate")
    else:  # Linux/Mac
        activate_cmd = "source .venv/bin/activate && pip install -r requirements.txt"
        print("\n📋 To activate the virtual environment manually:")
        print("   source .venv/bin/activate")
    
    # Install dependencies
    if not run_command(activate_cmd, "Installing dependencies"):
        print("\n⚠️  Automatic installation failed. Try manual installation:")
        print(f"   {activate_cmd}")
        return False
    
    print("\n🎉 Setup complete!")
    print("\n🚀 Quick start:")
    print("   1. Activate virtual environment (see above)")
    print("   2. cd scripts")
    print("   3. python cli.py --csv ../data/sample_products.csv --query 'blue lamp' --k 5")
    
    return True

if __name__ == "__main__":
    success = setup_environment()
    sys.exit(0 if success else 1)