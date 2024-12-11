#!/bin/bash

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Prerequisite Check Function
check_and_install_prerequisites() {
    echo -e "${YELLOW}[*] Checking System Prerequisites${NC}"
    
    # Prerequisites Checklist
    local prerequisites=(
        "python3.8"
        "python3.8-venv"
        "curl"
        "wget"
        "systemd"
    )
    
    local missing_packages=()
    
    # Check each prerequisite
    for pkg in "${prerequisites[@]}"; do
        if ! dpkg -s "$pkg" >/dev/null 2>&1; then
            missing_packages+=("$pkg")
        fi
    done
    
    # Install missing packages
    if [ ${#missing_packages[@]} -ne 0 ]; then
        echo -e "${RED}[!] Missing Packages Detected:${NC}"
        for missing in "${missing_packages[@]}"; do
            echo -e "${YELLOW}    - $missing${NC}"
        done
        
        echo -e "${YELLOW}[*] Updating Package Lists${NC}"
        sudo apt update
        
        echo -e "${YELLOW}[*] Installing Missing Packages${NC}"
        sudo apt install -y "${missing_packages[@]}"
    else
        echo -e "${GREEN}[✓] All Prerequisites Installed${NC}"
    fi
}

# System Compatibility Check
check_system_compatibility() {
    echo -e "${YELLOW}[*] Checking System Compatibility${NC}"
    
    # Check Ubuntu/Debian
    if [ ! -f /etc/os-release ]; then
        echo -e "${RED}[!] Unsupported Operating System${NC}"
        exit 1
    fi
    
    # Source OS information
    source /etc/os-release
    
    # Check if it's Ubuntu or Debian
    if [[ "$ID" != "ubuntu" && "$ID" != "debian" ]]; then
        echo -e "${RED}[!] Only Ubuntu and Debian are supported${NC}"
        exit 1
    fi
    
    # Check Python version
    if ! command -v python3.8 &> /dev/null; then
        echo -e "${YELLOW}[*] Python 3.8 not found. Attempting to install...${NC}"
        sudo add-apt-repository -y ppa:deadsnakes/ppa
        sudo apt update
        sudo apt install -y python3.8 python3.8-venv python3.8-dev
    fi
    
    echo -e "${GREEN}[✓] System Compatibility Verified${NC}"
}

# VPN Server Deployment Function
deploy_vpn_server() {
    echo -e "${YELLOW}[*] Starting VPN Server Deployment${NC}"
    
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
    
    echo -e "${GREEN}[✓] VPN Server Deployment Complete${NC}"
}

# Main Deployment Script
main() {
    echo -e "${YELLOW}[*] Quantum VPN Server Deployment Script${NC}"
    
    # Prerequisite and Compatibility Checks
    check_system_compatibility
    check_and_install_prerequisites
    
    # Deploy VPN Server
    deploy_vpn_server
    
    echo -e "${GREEN}[✓] Deployment Process Finished${NC}"
}

# Run Main Deployment
main
