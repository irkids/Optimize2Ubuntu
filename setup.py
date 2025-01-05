#!/usr/bin/env python3
import os
import sys
import subprocess
import venv

def install_system_dependencies():
    """Install required system dependencies"""
    try:
        subprocess.run(['apt-get', 'update'], check=True)
        system_packages = [
            'python3-dev',
            'python3-venv',
            'build-essential',
            'libpq-dev',
            'gcc',
            'git',
            'pkg-config',
            'libssl-dev'
        ]
        subprocess.run(['apt-get', 'install', '-y'] + system_packages, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing system dependencies: {e}")
        return False

def setup_virtual_environment():
    """Create and setup virtual environment"""
    try:
        venv_path = '/opt/script_venv'
        if not os.path.exists(venv_path):
            print("Creating virtual environment...")
            venv.create(venv_path, with_pip=True)
        
        venv_pip = os.path.join(venv_path, 'bin', 'pip')
        
        # Base packages
        base_packages = [
            'wheel',
            'setuptools>=45.0.0'
        ]
        
        # Main packages
        packages = [
            'alembic',  # DB migrations
            'asyncpg',
            'sqlalchemy[asyncio]',
            'fastapi',
            'uvicorn[standard]',
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
            'netifaces',
            'PyJWT',
            'python-jose[cryptography]',
            'ansible-runner',
            'PyYAML',
            'aiohttp',
            'websockets',
            'scapy',
            'pyroute2',
            'redis',
            'toml',
            'dynaconf',
            'loguru',
            'structlog',
            'numpy',
            'pandas',
            'scikit-learn',
            'tensorflow',
            'opentelemetry-api',
            'opentelemetry-sdk',
            'statsd',
            'elasticsearch',
            'aio-pika',
            'kafka-python',
            'nats-py',
            'boto3',
            'google-cloud-storage',
            'azure-identity',
            'aiocache'
        ]
        
        # Install base packages first
        for package in base_packages:
            print(f"Installing {package}...")
            subprocess.run([venv_pip, 'install', '--no-cache-dir', package], check=True)
        
        # Install main packages
        for package in packages:
            print(f"Installing {package}...")
            try:
                subprocess.run([venv_pip, 'install', '--no-cache-dir', package], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error installing {package}: {e}")
                continue
        
        # Create runner script
        runner_script = "run_main.sh"
        with open(runner_script, 'w') as f:
            f.write(f"""#!/bin/bash
source {os.path.join(venv_path, 'bin/activate')}
exec python3 "$@"
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

    if not install_system_dependencies():
        sys.exit(1)

    if not setup_virtual_environment():
        sys.exit(1)

    print("\nSetup complete! Run your script using: ./run_main.sh your_script.py")
