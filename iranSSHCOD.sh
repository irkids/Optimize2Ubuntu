#!/bin/bash

set -e

# Color Codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Log Function
log() {
    echo -e "${GREEN}[+] $1${NC}"
}

error() {
    echo -e "${RED}[!] Error: $1${NC}" >&2
    exit 1
}

# Prompt for Custom SSH Port
get_ssh_port() {
    read -p "Enter new SSH port (recommended between 49152-65535): " SSH_PORT
    if [[ ! "$SSH_PORT" =~ ^[0-9]+$ ]] || [ "$SSH_PORT" -lt 1024 ] || [ "$SSH_PORT" -gt 65535 ]; then
        error "Invalid port number. Must be between 1024-65535."
    fi
}

# DNS Configuration Function
configure_dns() {
    log "Configuring DNS Settings..."
    
    # DNS Provider Options
    echo "Select DNS Provider:"
    echo "1. Cloudflare (1.1.1.1)"
    echo "2. Google (8.8.8.8)"
    echo "3. Current System DNS"
    echo "4. Custom DNS"
    
    read -p "Enter your choice (1-4): " DNS_CHOICE
    
    case $DNS_CHOICE in
        1)
            # Cloudflare DNS
            cat << EOF > /etc/resolv.conf
nameserver 1.1.1.1
nameserver 1.0.0.1
nameserver 2606:4700:4700::1111
nameserver 2606:4700:4700::1001
EOF
            log "Configured Cloudflare DNS"
            ;;
        
        2)
            # Google DNS
            cat << EOF > /etc/resolv.conf
nameserver 8.8.8.8
nameserver 8.8.4.4
nameserver 2001:4860:4860::8888
nameserver 2001:4860:4860::8844
EOF
            log "Configured Google DNS"
            ;;
        
        3)
            # Keep current DNS
            log "Keeping current DNS configuration"
            ;;
        
        4)
            # Custom DNS
            read -p "Enter Primary DNS (IPv4): " PRIMARY_DNS
            read -p "Enter Secondary DNS (IPv4): " SECONDARY_DNS
            read -p "Enter Primary IPv6 DNS: " PRIMARY_IPV6_DNS
            read -p "Enter Secondary IPv6 DNS: " SECONDARY_IPV6_DNS
            
            cat << EOF > /etc/resolv.conf
nameserver $PRIMARY_DNS
nameserver $SECONDARY_DNS
nameserver $PRIMARY_IPV6_DNS
nameserver $SECONDARY_IPV6_DNS
EOF
            log "Configured Custom DNS"
            ;;
        
        *)
            error "Invalid DNS provider selection"
            ;;
    esac
    
    # Prevent DNS override
    chattr +i /etc/resolv.conf
}

# Advanced Kernel Network Optimization
kernel_network_optimization() {
    log "Applying advanced kernel network optimizations..."
    
    # Create backup of sysctl.conf
    cp /etc/sysctl.conf /etc/sysctl.conf.backup
    
    # Comprehensive Network Tuning
    cat << EOF > /etc/sysctl.conf
# IPv4 and IPv6 Networking
net.ipv4.ip_forward=1
net.ipv6.conf.all.forwarding=1

# TCP Optimization
net.core.default_qdisc=fq_codel
net.ipv4.tcp_congestion_control=bbr
net.ipv4.tcp_fastopen=3
net.core.somaxconn=65536
net.ipv4.tcp_max_syn_backlog=65536
net.ipv4.tcp_max_tw_buckets=1440000
net.ipv4.ip_local_port_range=1024 65535

# Connection Tracking
net.netfilter.nf_conntrack_max=524288
net.netfilter.nf_conntrack_tcp_timeout_established=28800

# IPv6 Specific Optimizations
net.ipv6.conf.default.disable_ipv6=0
net.ipv6.conf.all.disable_ipv6=0
net.ipv6.conf.all.accept_ra=2
net.ipv6.conf.default.accept_ra=2

# Security Enhancements
net.ipv4.conf.default.rp_filter=1
net.ipv4.conf.all.rp_filter=1
net.ipv4.tcp_syncookies=1
net.ipv4.tcp_rfc1337=1

# Memory Management
vm.swappiness=10
vm.overcommit_memory=1
vm.dirty_ratio=15
vm.dirty_background_ratio=5

# File System
fs.file-max=2097152
EOF

    # Apply sysctl settings
    sysctl -p
}

# Enhanced Performance Monitoring
performance_monitoring() {
    log "Installing advanced performance monitoring tools..."
    
    # Comprehensive Monitoring Suite
    apt-get install -y \
        htop \
        iotop \
        atop \
        glances \
        sysstat \
        netdata \
        telegraf \
        prometheus-node-exporter

    # Network Diagnostic Tools
    apt-get install -y \
        tcpdump \
        nethogs \
        iftop \
        bmon

    # Performance Logging
    systemctl enable sysstat
    systemctl start sysstat

    # Netdata Real-time Monitoring
    systemctl enable netdata
    systemctl start netdata
}

# Security and Hardening
advanced_security_hardening() {
    log "Implementing advanced security hardening..."
    
    # Install Security Tools
    apt-get install -y \
        fail2ban \
        rkhunter \
        chkrootkit \
        libpam-tmpdir \
        needrestart

    # Configure Fail2Ban
    cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local
    sed -i 's/bantime  = 10m/bantime  = 1h/' /etc/fail2ban/jail.local
    sed -i 's/maxretry = 5/maxretry = 3/' /etc/fail2ban/jail.local

    # Enable automatic security updates
    dpkg-reconfigure -plow unattended-upgrades

    # Disable core dumps
    echo "* soft core 0" >> /etc/security/limits.conf
    echo "* hard core 0" >> /etc/security/limits.conf

    # Harden SSH Configuration
    sed -i 's/^#MaxAuthTries 6/MaxAuthTries 3/' /etc/ssh/sshd_config
    sed -i 's/^#AllowUsers/AllowUsers/' /etc/ssh/sshd_config
}

# System Update
system_update() {
    log "Updating system packages..."
    apt-get update && apt-get upgrade -y
}

# Development Environment Setup
dev_environment() {
    log "Setting up comprehensive development environment..."

    # Python Setup
    apt-get install -y python3 python3-pip python3-venv
    pip3 install pytest psycopg2 ansible
    pip install psycopg2 

    # Node.js and NPM
    curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -
    apt-get install -y nodejs
    npm install -g npm@latest

    # Java
    apt-get install -y openjdk-11-jdk

    # Docker
    curl -fsSL https://get.docker.com | sh
    usermod -aG docker $USER
}

# Database Setup
database_setup() {
    log "Installing and configuring databases..."
    
    # PostgreSQL
    apt-get install -y postgresql postgresql-contrib
    
    # Redis
    apt-get install -y redis-server

    # Elasticsearch, Logstash, Kibana
    wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | apt-key add -
    echo "deb https://artifacts.elastic.co/packages/7.x/apt stable main" | tee /etc/apt/sources.list.d/elastic-7.x.list
    apt-get update
    apt-get install -y elasticsearch logstash kibana
}

# Web and Infrastructure Tools
web_infrastructure_setup() {
    log "Setting up web and infrastructure tools..."

    # Nginx
    apt-get install -y nginx

    # Kubernetes and Terraform
    snap install kubectl --classic
    wget https://releases.hashicorp.com/terraform/1.5.7/terraform_1.5.7_linux_amd64.zip
    unzip terraform_1.5.7_linux_amd64.zip
    mv terraform /usr/local/bin/

    # Jenkins
    wget -q -O - https://pkg.jenkins.io/debian-stable/jenkins.io.key | apt-key add -
    sh -c 'echo deb https://pkg.jenkins.io/debian-stable binary/ > /etc/apt/sources.list.d/jenkins.list'
    apt-get update
    apt-get install -y jenkins

    # HashiCorp Vault
    wget https://releases.hashicorp.com/vault/1.14.0/vault_1.14.0_linux_amd64.zip
    unzip vault_1.14.0_linux_amd64.zip
    mv vault /usr/local/bin/
}

# Monitoring and Security
monitoring_security_setup() {
    log "Configuring monitoring and security tools..."

    # Prometheus and Grafana
    apt-get install -y prometheus grafana

    # AWS Secrets Manager CLI
    pip3 install awscli
}

# Custom SSH Port Configuration
configure_ssh() {
    log "Configuring custom SSH port..."
    get_ssh_port
    
    sed -i "s/^#*Port.*/Port $SSH_PORT/" /etc/ssh/sshd_config
    
    # Update UFW to allow new SSH port
    ufw allow $SSH_PORT/tcp
    ufw delete allow 22/tcp
    
    systemctl restart ssh
    systemctl restart sshd
}

# Frontend Development Setup
frontend_setup() {
    log "Setting up frontend development environment..."
    
    npm install -g create-react-app
    npm install -g \
        react \
        react-router-dom \
        redux \
        axios \
        @mui/material \
        tailwindcss \
        socket.io-client \
        react-toastify \
        formik \
        yup \
        react-query \
        dayjs \
        styled-components \
        react-icons \
        jest \
        @testing-library/react \
        react-i18next \
        react-intl \
        moment-jalaali \
        recharts \
        apexcharts
}

# SSH Optimization
optimize_ssh() {
    log "Optimizing SSH configuration..."

    local SSH_CONFIG="/etc/ssh/sshd_config"

    # Disable DNS lookup
    sed -i '/^#*UseDNS/s/.*/UseDNS no/' $SSH_CONFIG

    # Client Alive Settings
    sed -i '/^#*ClientAliveInterval/s/.*/ClientAliveInterval 60/' $SSH_CONFIG
    sed -i '/^#*ClientAliveCountMax/s/.*/ClientAliveCountMax 3/' $SSH_CONFIG

    # Connection Limits
    sed -i '/^#*MaxSessions/s/.*/MaxSessions 10/' $SSH_CONFIG
    sed -i '/^#*MaxStartups/s/.*/MaxStartups 10:30:100/' $SSH_CONFIG

    # Security Protocol
    sed -i '/^#*Protocol/s/.*/Protocol 2/' $SSH_CONFIG
    sed -i '/^#*Compression/s/.*/Compression yes/' $SSH_CONFIG

    # Cipher and Encryption Optimizations
    sed -i '/^#*Ciphers/s/.*/Ciphers aes128-ctr,aes192-ctr,aes256-ctr/' $SSH_CONFIG
    sed -i '/^#*MACs/s/.*/MACs hmac-sha2-256,hmac-sha2-512/' $SSH_CONFIG
    sed -i '/^#*KexAlgorithms/s/.*/KexAlgorithms diffie-hellman-group14-sha256/' $SSH_CONFIG

    # Custom SSH Port
    configure_ssh

    # Restart SSH Service
    systemctl restart sshd
}

# External Optimization
run_external_optimization() {
    log "Running external SSH optimization script..."
    bash <(curl -Ls https://raw.githubusercontent.com/irkids/IR_OptimizeR/refs/heads/main/sshTCP.sh --ipv4)
}

# VPN Protocol Prerequisites
vpn_prerequisites() {
    log "Installing VPN protocol prerequisites and dependencies..."

    # System-wide dependencies
    apt-get install -y \
        build-essential \
        libssl-dev \
        libz-dev \
        libreadline-dev \
        libnss3-dev \
        libnspr4-dev \
        libpcre3-dev \
        libcurl4-openssl-dev \
        libpcap-dev \
        pkg-config \
        software-properties-common

    # Network and Security Libraries
    apt-get install -y \
        libsodium-dev \
        libmbedtls-dev \
        libgnutls28-dev \
        liblz4-dev \
        libprotobuf-dev \
        protobuf-compiler

    # Kernel modules and headers
    apt-get install -y \
        linux-headers-$(uname -r) \
        linux-modules-extra-$(uname -r)

    # Cryptography and Security Tools
    apt-get install -y \
        openssl \
        libcap2-bin \
        iptables \
        ufw \
        libmnl0 \
        libelf1

    # Network Utility Tools
    apt-get install -y \
        iproute2 \
        net-tools \
        ethtool \
        iputils-ping \
        mtr-tiny

    # Performance and Monitoring Tools
    apt-get install -y \
        iperf3

    # Load necessary kernel modules
    modprobe udp_tunnel
    modprobe ip_tunnel
    modprobe tun
    modprobe esp4

    # Enable IP forwarding
    echo "net.ipv4.ip_forward=1" >> /etc/sysctl.conf
    echo "net.ipv6.conf.all.forwarding=1" >> /etc/sysctl.conf
    sysctl -p
}

# Main Execution
main() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root. Use sudo."
    fi

    log "Starting full server and SSH optimization setup..."
    system_update
    dev_environment
    database_setup
    web_infrastructure_setup
    monitoring_security_setup
    frontend_setup
    optimize_ssh
    vpn_prerequisites
    configure_dns
    kernel_network_optimization
    performance_monitoring
    advanced_security_hardening
    run_external_optimization

    log "All tasks completed successfully!"
}

# Run main function
main

exit 0
