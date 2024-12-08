#!/usr/bin/env bash

# Enhanced SSH Security and WebSocket Proxy Setup Script
# Specifically designed for Ubuntu 20.04+
# Requires root privileges

# Strict error handling
set -euo pipefail

# Global configurations
readonly SCRIPT_VERSION="1.1.0"
readonly CONFIG_DIR="/etc/ssh_enhanced"
readonly LOG_FILE="/var/log/ssh_enhanced.log"
readonly METRICS_DIR="/var/lib/ssh_metrics"

# ANSI Color Codes for Enhanced Logging
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m' # No Color

# Comprehensive Logging Function
log() {
    local level="${1^^}"
    local message="$2"
    local color=""

    case "$level" in
        "ERROR")   color="$RED" ;;
        "WARNING") color="$YELLOW" ;;
        "INFO")    color="$GREEN" ;;
        *)         color="$NC" ;;
    esac

    printf "[%s] [%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$level" "$message" | tee -a "$LOG_FILE"
    printf "${color}[%s] [%s] %s${NC}\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$level" "$message"
}

# Ubuntu Version Compatibility Check
check_ubuntu_version() {
    local version=$(grep -oP '20\.04|22\.04|24\.04' /etc/os-release)
    if [[ -z "$version" ]]; then
        log "ERROR" "This script is only compatible with Ubuntu 20.04, 22.04, or 24.04"
        exit 1
    fi
    log "INFO" "Compatible Ubuntu version detected: $version"
}

# Comprehensive Package Installation with Error Handling
install_packages() {
    local packages=("$@")
    
    # Update package lists
    apt-get update || {
        log "ERROR" "Failed to update package lists"
        return 1
    }

    # Install packages with detailed error reporting
    DEBIAN_FRONTEND=noninteractive apt-get install -y "${packages[@]}" || {
        local failed_packages=()
        for pkg in "${packages[@]}"; do
            if ! apt-get install -y "$pkg"; then
                failed_packages+=("$pkg")
            fi
        done

        if [[ ${#failed_packages[@]} -gt 0 ]]; then
            log "ERROR" "Failed to install packages: ${failed_packages[*]}"
            return 1
        fi
    }

    log "INFO" "Successfully installed packages: ${packages[*]}"
}

# Prometheus Node Exporter Installation
install_node_exporter() {
    local version="1.6.1"
    local arch=$(dpkg --print-architecture)
    local download_url="https://github.com/prometheus/node_exporter/releases/download/v${version}/node_exporter-${version}.linux-${arch}.tar.gz"

    # Create temporary directory
    local temp_dir=$(mktemp -d)
    cd "$temp_dir"

    # Download and extract
    wget -O node_exporter.tar.gz "$download_url"
    tar xvfz node_exporter.tar.gz

    # Move binary
    mv "node_exporter-${version}.linux-${arch}/node_exporter" /usr/local/bin/
    chmod +x /usr/local/bin/node_exporter

    # Create systemd service
    cat > /etc/systemd/system/node_exporter.service << EOL
[Unit]
Description=Node Exporter
Wants=network-online.target
After=network-online.target

[Service]
User=root
ExecStart=/usr/local/bin/node_exporter

[Install]
WantedBy=default.target
EOL

    # Enable and start service
    systemctl daemon-reload
    systemctl enable node_exporter
    systemctl start node_exporter

    log "INFO" "Node Exporter ${version} installed successfully"

    # Clean up
    cd /
    rm -rf "$temp_dir"
}

# Go Installation with Secure Verification
install_go() {
    local go_version="1.21.6"
    local go_arch="linux-amd64"
    local go_url="https://golang.org/dl/go${go_version}.${go_arch}.tar.gz"
    local go_checksum_url="https://golang.org/dl/go${go_version}.${go_arch}.tar.gz.sha256"

    # Download and verify checksum
    local temp_dir=$(mktemp -d)
    cd "$temp_dir"

    wget "$go_url"
    wget "$go_checksum_url"

    # Verify download integrity
    if ! sha256sum -c <(awk '{print $1, " go'${go_version}.${go_arch}.tar.gz'"}' go${go_version}.${go_arch}.tar.gz.sha256); then
        log "ERROR" "Go download checksum verification failed"
        return 1
    fi

    # Remove existing Go installation
    rm -rf /usr/local/go

    # Install Go
    tar -C /usr/local -xzf go${go_version}.${go_arch}.tar.gz

    # Set up environment
    mkdir -p /usr/local/go/workspace
    export GOPATH="/usr/local/go/workspace"
    export PATH="/usr/local/go/bin:$PATH"

    # Verify installation
    go version

    log "INFO" "Go ${go_version} installed successfully"

    # Clean up
    cd /
    rm -rf "$temp_dir"
}

# Enhanced WebSocket Proxy with Improved Security
create_websocket_proxy() {
    export GOPATH="/usr/local/go/workspace"
    mkdir -p "$GOPATH/src/ssh_websocket_proxy"
    cd "$GOPATH/src/ssh_websocket_proxy"

    # More secure WebSocket proxy implementation
    cat > proxy.go << 'EOL'
package main

import (
    "crypto/tls"
    "log"
    "net"
    "net/http"
    "sync"
    "time"

    "github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
    ReadBufferSize:  32768,  // Increased buffer size
    WriteBufferSize: 32768,
    CheckOrigin: func(r *http.Request) bool {
        // Implement stricter origin checking
        return r.Header.Get("Origin") == "https://localhost"
    },
    EnableCompression: true, // Enable compression
}

func proxySSH(ws *websocket.Conn, sshConn net.Conn) {
    var wg sync.WaitGroup
    wg.Add(2)

    // Concurrent read/write with timeout and error handling
    go func() {
        defer wg.Done()
        defer ws.Close()
        defer sshConn.Close()

        ws.SetReadDeadline(time.Now().Add(10 * time.Minute))
        
        for {
            _, data, err := ws.ReadMessage()
            if err != nil {
                log.Printf("WebSocket read error: %v", err)
                break
            }
            _, err = sshConn.Write(data)
            if err != nil {
                log.Printf("SSH write error: %v", err)
                break
            }
        }
    }()

    go func() {
        defer wg.Done()
        defer ws.Close()
        defer sshConn.Close()

        buffer := make([]byte, 32768)
        for {
            n, err := sshConn.Read(buffer)
            if err != nil {
                log.Printf("SSH read error: %v", err)
                break
            }
            err = ws.WriteMessage(websocket.BinaryMessage, buffer[:n])
            if err != nil {
                log.Printf("WebSocket write error: %v", err)
                break
            }
        }
    }()

    wg.Wait()
}

func main() {
    http.HandleFunc("/ssh", func(w http.ResponseWriter, r *http.Request) {
        ws, err := upgrader.Upgrade(w, r, nil)
        if err != nil {
            log.Printf("Upgrade error: %v", err)
            return
        }

        sshConn, err := net.DialTimeout("tcp", "127.0.0.1:22", 5*time.Second)
        if err != nil {
            log.Printf("SSH connection error: %v", err)
            ws.Close()
            return
        }

        proxySSH(ws, sshConn)
    })

    // TLS Configuration with Modern Cipher Suites
    tlsConfig := &tls.Config{
        MinVersion: tls.VersionTLS13,
        CipherSuites: []uint16{
            tls.TLS_AES_256_GCM_SHA384,
            tls.TLS_CHACHA20_POLY1305_SHA256,
        },
        PreferServerCipherSuites: true,
    }

    server := &http.Server{
        Addr:         ":8080",
        TLSConfig:    tlsConfig,
        ReadTimeout:  10 * time.Second,
        WriteTimeout: 10 * time.Second,
    }

    log.Println("Starting WebSocket SSH proxy on :8080 with enhanced security")
    log.Fatal(server.ListenAndServeTLS("/etc/ssl/private/server.crt", "/etc/ssl/private/server.key"))
}
EOL

    # Go module initialization
    go mod init ssh_websocket_proxy
    go mod tidy
    go get github.com/gorilla/websocket

    # Compile with additional security flags
    go build -ldflags="-s -w" -o /usr/local/bin/ssh_websocket_proxy proxy.go

    log "INFO" "Enhanced WebSocket proxy compiled with advanced security"
}

# SSH Configuration with Ultra-Secure Settings
configure_ssh() {
    local ssh_config="/etc/ssh/sshd_config"
    
    # Backup existing configuration
    cp "$ssh_config" "${ssh_config}.backup.$(date +%Y%m%d_%H%M%S)"
    
    # Ultra-Enhanced SSH Security Configuration
    cat > "$ssh_config" << EOL
# Ultra-Secure SSH Configuration

# Basic Settings
Port 22
AddressFamily inet
ListenAddress 0.0.0.0

# Authentication Hardening
PermitRootLogin prohibit-password
PasswordAuthentication no
ChallengeResponseAuthentication no
MaxAuthTries 3
LoginGraceTime 30s

# Key Authentication
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys
AuthenticationMethods publickey

# Advanced Security
HostKeyAlgorithms ssh-ed25519,rsa-sha2-512,rsa-sha2-256
KexAlgorithms curve25519-sha256@libssh.org,diffie-hellman-group-exchange-sha256
Ciphers chacha20-poly1305@openssh.com,aes256-gcm@openssh.com
MACs hmac-sha2-512-etm@openssh.com,hmac-sha2-256-etm@openssh.com

# Strict Access Control
AllowUsers *@localhost
DenyUsers root

# Logging and Monitoring
LogLevel VERBOSE
MaxSessions 3
ClientAliveInterval 300
ClientAliveCountMax 0

# Disable Unnecessary Features
X11Forwarding no
AllowAgentForwarding no
AllowTcpForwarding no
PermitTunnel no
PermitEmptyPasswords no

# Additional Protections
StrictModes yes
UsePrivilegeSeparation sandbox
EOL

    # Restart SSH with enhanced security
    systemctl restart ssh
    log "INFO" "SSH configuration updated with ultra-secure settings"
}

# Prometheus Monitoring Setup
setup_prometheus() {
    # Create Prometheus configuration directory
    mkdir -p /etc/prometheus

    # Prometheus Configuration
    cat > /etc/prometheus/prometheus.yml << EOL
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'node_exporter'
    static_configs:
      - targets: ['localhost:9100']

  - job_name: 'ssh_proxy'
    static_configs:
      - targets: ['localhost:8080']
EOL

    # Install latest Prometheus
    local prometheus_version="2.48.1"
    local arch=$(dpkg --print-architecture)
    local download_url="https://github.com/prometheus/prometheus/releases/download/v${prometheus_version}/prometheus-${prometheus_version}.linux-${arch}.tar.gz"

    local temp_dir=$(mktemp -d)
    cd "$temp_dir"

    wget -O prometheus.tar.gz "$download_url"
    tar xvfz prometheus.tar.gz

    # Install Prometheus binary
    cp "prometheus-${prometheus_version}.linux-${arch}/prometheus" /usr/local/bin/
    cp "prometheus-${prometheus_version}.linux-${arch}/promtool" /usr/local/bin/

    # Create systemd service
    cat > /etc/systemd/system/prometheus.service << EOL
[Unit]
Description=Prometheus Monitoring
Wants=network-online.target
After=network-online.target

[Service]
ExecStart=/usr/local/bin/prometheus --config.file=/etc/prometheus/prometheus.yml \
  --storage.tsdb.path=/var/lib/prometheus \
  --web.console.templates=/etc/prometheus/consoles \
  --web.console.libraries=/etc/prometheus/console_libraries

[Install]
WantedBy=default.target
EOL

    # Create directories and set permissions
    mkdir -p /var/lib/prometheus
    systemctl daemon-reload
    systemctl enable prometheus
    systemctl start prometheus

    log "INFO" "Prometheus ${prometheus_version} installed and configured"

    # Clean up
    cd /
    rm -rf "$temp_dir"
}

# Firewall Configuration
configure_firewall() {
    # Install UFW if not present
    apt-get update
    apt-get install -y ufw

    # Reset and enable UFW
    ufw --force reset
    
    # Allow SSH, WebSocket Proxy, and Monitoring Ports
    ufw allow 22/tcp    # SSH
    ufw allow 8080/tcp  # WebSocket Proxy
    ufw allow 9100/tcp  # Node Exporter
    ufw allow 9090/tcp  # Prometheus

    # Enable UFW with default deny policy
    ufw default deny incoming
    ufw default allow outgoing
    ufw --force enable

    log "INFO" "Firewall configured with strict rules"
}

# Generate Self-Signed Certificate Function
generate_self_signed_certificate() {
    local cert_dir="/etc/ssl/private"
    
    # Ensure OpenSSL is installed
    install_packages "openssl"
    
    # Create directory if not exists
    mkdir -p "$cert_dir"
    
    # Generate self-signed certificate
    openssl req -x509 -nodes -days 365 -newkey rsa:4096 \
        -keyout "$cert_dir/server.key" \
        -out "$cert_dir/server.crt" \
        -subj "/C=US/ST=NetworkSecurity/L=SSHEnhanced/O=LocalDevelopment/CN=localhost"
    
    # Set proper permissions
    chmod 600 "$cert_dir/server.key"
    
    log "INFO" "Self-signed certificates generated"
}

# Main Installation Function
main() {
    # Ensure root privileges
    if [[ $EUID -ne 0 ]]; then
        log "ERROR" "This script must be run as root. Use sudo."
        exit 1
    fi

    # Check Ubuntu Version
    check_ubuntu_version

    # Essential package installation
    local base_packages=(
        "openssh-server" "fail2ban" "ufw"
        "git" "curl" "wget" "software-properties-common"
        "build-essential"
    )

    # Install base packages
    install_packages "${base_packages[@]}"

    # Create configuration and metrics directories
    mkdir -p "$CONFIG_DIR"
    mkdir -p "$METRICS_DIR"

    # Install Node Exporter
    install_node_exporter

    # Install Go
    install_go

    # Generate self-signed certificates
    generate_self_signed_certificate

    # Create WebSocket Proxy
    create_websocket_proxy

    # Configure SSH
    configure_ssh

    # Setup Prometheus Monitoring
    setup_prometheus

    # Configure Firewall
    configure_firewall

    # Log successful completion
    log "INFO" "SSH Enhanced Security Setup Complete (Version ${SCRIPT_VERSION})"
}

# Execute the main function
main
