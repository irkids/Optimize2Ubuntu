#!/usr/bin/env bash

# Prerequisites and Dependency Installation Script
# Supports Ubuntu/Debian and CentOS/RHEL

# Strict error handling
set -euo pipefail

# Global configurations
readonly CONFIG_DIR="/etc/ssh_enhanced"
readonly LOG_FILE="/var/log/ssh_enhanced.log"
readonly METRICS_DIR="/var/lib/ssh_metrics"

# Logging function with enhanced formatting
log() {
    local level="${1^^}"
    local message="$2"
    printf "[%s] [%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$level" "$message" | tee -a "$LOG_FILE"
}

# Detect package manager and OS
detect_package_manager() {
    if command -v apt-get &> /dev/null; then
        echo "apt"
    elif command -v yum &> /dev/null; then
        echo "yum"
    else
        log "ERROR" "Unsupported package manager"
        return 1
    fi
}

# Centralized package management
install_packages() {
    local package_manager
    package_manager=$(detect_package_manager)
    local packages=("$@")

    if [ "$package_manager" == "apt" ]; then
        DEBIAN_FRONTEND=noninteractive apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y "${packages[@]}"
    elif [ "$package_manager" == "yum" ]; then
        yum install -y "${packages[@]}"
    fi
}

# Install Go with version management
install_go() {
    local go_version="1.21.5"
    local go_arch="linux-amd64"
    local go_url="https://golang.org/dl/go${go_version}.${go_arch}.tar.gz"
    
    # Remove existing Go installation if exists
    rm -rf /usr/local/go

    # Download and install Go
    curl -L "$go_url" | tar -C /usr/local -xzf -
    
    # Update PATH and environment
    export PATH="/usr/local/go/bin:$PATH"
    
    # Verify Go installation
    go version
    
    log "INFO" "Go $go_version installed successfully"
}

# Install Gorilla WebSocket library
install_websocket_library() {
    mkdir -p "$GOPATH/src/github.com/gorilla"
    cd "$GOPATH/src/github.com/gorilla"
    
    # Clone Gorilla WebSocket library
    git clone https://github.com/gorilla/websocket.git
    
    # Verify library installation
    cd websocket
    go mod init
    go mod tidy
    
    log "INFO" "Gorilla WebSocket library installed"
}

# Secure certificate generation (self-signed for simplicity)
generate_self_signed_certificate() {
    local cert_dir="/etc/ssl/private"
    
    # Ensure OpenSSL is installed
    install_packages "openssl"
    
    # Create directory if not exists
    mkdir -p "$cert_dir"
    
    # Generate self-signed certificate
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout "$cert_dir/server.key" \
        -out "$cert_dir/server.crt" \
        -subj "/C=US/ST=NetworkSecurity/L=SSHEnhanced/O=LocalDevelopment/CN=localhost"
    
    # Set proper permissions
    chmod 600 "$cert_dir/server.key"
    
    log "INFO" "Self-signed certificates generated"
}

# Optimized WebSocket proxy (Go implementation)
create_websocket_proxy() {
    # Ensure Go is installed
    export GOPATH="${GOPATH:-/usr/local/go/workspace}"
    mkdir -p "$GOPATH"
    
    cat > "$GOPATH/ssh_websocket_proxy.go" << 'EOL'
package main

import (
    "log"
    "net"
    "net/http"
    "sync"

    "github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
    ReadBufferSize:  4096,
    WriteBufferSize: 4096,
    CheckOrigin: func(r *http.Request) bool {
        // Note: In production, implement proper origin checking
        return true
    },
}

func proxySSH(ws *websocket.Conn, sshConn net.Conn) {
    var wg sync.WaitGroup
    wg.Add(2)

    go func() {
        defer wg.Done()
        defer ws.Close()
        defer sshConn.Close()

        for {
            _, data, err := ws.ReadMessage()
            if err != nil {
                break
            }
            _, err = sshConn.Write(data)
            if err != nil {
                break
            }
        }
    }()

    go func() {
        defer wg.Done()
        defer ws.Close()
        defer sshConn.Close()

        buffer := make([]byte, 4096)
        for {
            n, err := sshConn.Read(buffer)
            if err != nil {
                break
            }
            err = ws.WriteMessage(websocket.BinaryMessage, buffer[:n])
            if err != nil {
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
            log.Println(err)
            return
        }

        sshConn, err := net.Dial("tcp", "127.0.0.1:22")
        if err != nil {
            log.Println(err)
            ws.Close()
            return
        }

        proxySSH(ws, sshConn)
    })

    log.Println("Starting WebSocket SSH proxy on :8080")
    log.Fatal(http.ListenAndServeTLS(":8080", "/etc/ssl/private/server.crt", "/etc/ssl/private/server.key", nil))
}
EOL

    # Prepare Go modules
    cd "$GOPATH"
    go mod init ssh_websocket_proxy
    go get github.com/gorilla/websocket

    # Compile Go WebSocket proxy
    go build -o /usr/local/bin/ssh_websocket_proxy "$GOPATH/ssh_websocket_proxy.go"
    log "INFO" "WebSocket proxy compiled"
}

# Configure SSH with enhanced security
configure_ssh() {
    local ssh_config="/etc/ssh/sshd_config"
    
    # Backup existing configuration
    cp "$ssh_config" "${ssh_config}.backup"
    
    # Enhance SSH security
    cat > "$ssh_config" << EOL
# Enhanced SSH Configuration
Port 22
AddressFamily inet
ListenAddress 0.0.0.0

# Authentication
PermitRootLogin no
PasswordAuthentication no
ChallengeResponseAuthentication no

# Key-based authentication
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys

# Additional security
MaxAuthTries 3
LoginGraceTime 30

# Logging
LogLevel VERBOSE

# Disable empty passwords
PermitEmptyPasswords no

# Only use strong encryption
Ciphers chacha20-poly1305@openssh.com,aes256-gcm@openssh.com
MACs hmac-sha2-512-etm@openssh.com,hmac-sha2-256-etm@openssh.com
KexAlgorithms curve25519-sha256@libssh.org,diffie-hellman-group-exchange-sha256
EOL

    # Restart SSH service
    systemctl restart sshd
    log "INFO" "SSH configuration updated with enhanced security"
}

# Set up monitoring and logging
setup_monitoring() {
    # Ensure Prometheus and related tools are installed
    install_packages "prometheus" "node-exporter"
    
    # Create metrics directory
    mkdir -p "$METRICS_DIR"
    
    # Configure Prometheus
    cat > /etc/prometheus/prometheus.yml << EOL
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']
  
  - job_name: 'ssh_proxy'
    static_configs:
      - targets: ['localhost:8080']
EOL

    # Restart monitoring services
    systemctl restart prometheus
    systemctl restart node-exporter
    
    log "INFO" "Monitoring and logging setup complete"
}

# Prerequisites installation function
install_prerequisites() {
    # Ensure script is run with root privileges
    if [[ $EUID -ne 0 ]]; then
        log "ERROR" "This script must be run as root"
        exit 1
    fi

    # Essential packages
    local base_packages=(
        "openssh-server" "fail2ban" "ufw"
        "prometheus" "node-exporter"
        "git" "curl" "wget"
    )

    # Install base packages
    install_packages "${base_packages[@]}"
    
    # Install Go
    install_go
    
    # Install Gorilla WebSocket library
    install_websocket_library
    
    # Generate self-signed certificates
    generate_self_signed_certificate
}

# Main setup function
main() {
    # Install prerequisites
    install_prerequisites
    
    # Create optimized WebSocket proxy
    create_websocket_proxy
    
    # Configure SSH with enhanced security
    configure_ssh
    
    # Set up monitoring and logging
    setup_monitoring
    
    log "INFO" "SSH Enhanced Setup Complete"
}

# Execute main function
main
