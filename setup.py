#!/usr/bin/env python3
import os
import sys
import subprocess
import venv

def setup_virtual_environment():
    """Create and setup virtual environment"""
    try:
        venv_path = '/opt/script_venv'
        
        # Create virtual environment if it doesn't exist
        if not os.path.exists(venv_path):
            print("Creating virtual environment...")
            venv.create(venv_path, with_pip=True)
        
        # Get paths
        venv_pip = os.path.join(venv_path, 'bin', 'pip')
        
        # Install packages
        packages = [
            'wheel',
            'setuptools>=45.0.0',
            'asyncpg',
            'sqlalchemy',
            'fastapi',
            'uvicorn',
            'psutil',
            'prometheus_client',
            'kubernetes',
            'docker',
            'pytest',
            'pytest-asyncio',
            'hypothesis',
            'aioredis',
            'cryptography',
            'bcrypt',
            'passlib',
            'pydantic',
            'netifaces'
        ]
        
        for package in packages:
            print(f"Installing {package}...")
            subprocess.run([venv_pip, 'install', '--no-cache-dir', package], check=True)
        
        # Create runner script
        runner_script = os.path.join(os.getcwd(), "run_main.sh")
        with open(runner_script, 'w') as f:
            f.write(f"""#!/bin/bash
source {os.path.join(venv_path, 'bin/activate')}
python "$@"
""")
        
        os.chmod(runner_script, 0o755)
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if os.geteuid() != 0:
        print("This script must be run as root!")
        sys.exit(1)

    if not setup_virtual_environment():
        sys.exit(1)
    
    print("\nSetup complete! Run your script using: ./run_main.sh your_script.py")
