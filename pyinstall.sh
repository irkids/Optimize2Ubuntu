#!/bin/bash

# Create virtual environment
python3 -m venv vpn_server_env

# Activate virtual environment
source vpn_server_env/bin/activate

# Install dependencies
pip install uvloop psutil tensorflow numpy pandas networkx cryptography requests

# Create directory for VPN server
mkdir -p /opt/vpn_server

# Download the script directly from GitHub
curl -L https://raw.githubusercontent.com/irkids/Optimize2Ubuntu/refs/heads/main/iranssh.py -o /opt/vpn_server/iranssh.py

# Set up systemd service for auto-start
cat << EOF > /etc/systemd/system/quantum-vpn.service
[Unit]
Description=Quantum Adaptive VPN Server
After=network.target

[Service]
Type=simple
User=nobody
WorkingDirectory=/opt/vpn_server
ExecStart=/opt/vpn_server_env/bin/python3 /opt/vpn_server/iranssh.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
systemctl enable quantum-vpn
systemctl start quantum-vpn
