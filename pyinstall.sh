#!/bin/bash

# Update package lists
sudo apt update

# Install Python 3.8 venv package
sudo apt install -y python3.8-venv

# Create virtual environment
python3.8 -m venv vpn_server_env

# Activate virtual environment
source vpn_server_env/bin/activate

# Ensure pip is upgraded
pip install --upgrade pip

# Install dependencies
pip install uvloop psutil tensorflow numpy pandas networkx cryptography requests

# Create directory for VPN server
sudo mkdir -p /opt/vpn_server

# Download the script directly from GitHub
sudo curl -L https://raw.githubusercontent.com/irkids/Optimize2Ubuntu/refs/heads/main/iranssh.py -o /opt/vpn_server/iranssh.py

# Set permissions
sudo chown nobody:nogroup /opt/vpn_server/iranssh.py
sudo chmod 755 /opt/vpn_server/iranssh.py

# Set up systemd service for auto-start
sudo tee /etc/systemd/system/quantum-vpn.service > /dev/null << EOF
[Unit]
Description=Quantum Adaptive VPN Server
After=network.target

[Service]
Type=simple
User=nobody
WorkingDirectory=/opt/vpn_server
ExecStart=/root/vpn_server_env/bin/python3 /opt/vpn_server/iranssh.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd to recognize new service
sudo systemctl daemon-reload

# Enable and start the service
sudo systemctl enable quantum-vpn
sudo systemctl start quantum-vpn
