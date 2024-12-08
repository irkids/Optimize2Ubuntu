#!/usr/bin/env bash

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

# Centralized package management
install_packages() {
    local packages=("$@")
    if command -v apt-get &> /dev/null; then
        DEBIAN_FRONTEND=noninteractive apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y "${packages[@]}"
    elif command -v yum &> /dev/null; then
        yum install -y "${packages[@]}"
    else
        log "ERROR" "Unsupported package manager"
        return 1
    fi
}

# Secure certificate generation
generate_certificate() {
    local domain="$1"
    local cert_dir="/etc/ssl/private"
    
    # Ensure certbot is installed
    install_packages "certbot"
    
    # Use Certbot for Let's Encrypt certificates
    certbot certonly --standalone -d "$domain" --non-interactive --agree-tos --register-unsafely-without-email
    
    # Symlink generated certificates
    mkdir -p "$cert_dir"
    ln -sf "/etc/letsencrypt/live/$domain/fullchain.pem" "$cert_dir/server.crt"
    ln -sf "/etc/letsencrypt/live/$domain/privkey.pem" "$cert_dir/server.key"
    
    chmod 600 "$cert_dir/server.key"
    log "INFO" "Certificates generated for $domain"
}

# Optimized WebSocket proxy (Go implementation)
create_websocket_proxy() {
    # Ensure Go is installed
    install_packages "golang"

    cat > /usr/local/bin/ssh_websocket_proxy.go << 'EOL'
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
    cd /usr/local/bin
    go mod init ssh_websocket_proxy
    go get github.com/gorilla/websocket

    # Compile Go WebSocket proxy
    go build -o /usr/local/bin/ssh_websocket_proxy ssh_websocket_proxy.go
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
AllowUsers ${ALLOWED_SSH_USERS:-}

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

# Main setup function
main() {
    # Check if domain is provided
    if [ $# -eq 0 ]; then
        log "ERROR" "Domain argument is required"
        echo "Usage: $0 <domain>"
        exit 1
    fi

    local domain="$1"

    # Essential packages
    local base_packages=(
        "openssh-server" "fail2ban" "ufw"
        "prometheus" "node-exporter" "certbot"
        "golang" "golang-go"
    )

    # Ensure script is run with root privileges
    if [[ $EUID -ne 0 ]]; then
        log "ERROR" "This script must be run as root"
        exit 1
    fi

    # Update package list and install base packages
    install_packages "${base_packages[@]}"
    
    # Generate secure certificates
    generate_certificate "$domain"
    
    # Create optimized WebSocket proxy
    create_websocket_proxy
    
    # Configure SSH with enhanced security
    configure_ssh
    
    # Set up monitoring and logging
    setup_monitoring
    
    log "INFO" "SSH Enhanced Setup Complete for $domain"
}

# Execute main function
main "$@"
