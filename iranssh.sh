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
    
    # Use Certbot for Let's Encrypt certificates
    certbot certonly --standalone -d "$domain"
    
    # Symlink generated certificates
    ln -sf "/etc/letsencrypt/live/$domain/fullchain.pem" "$cert_dir/server.crt"
    ln -sf "/etc/letsencrypt/live/$domain/privkey.pem" "$cert_dir/server.key"
    
    chmod 600 "$cert_dir/server.key"
}

# Optimized WebSocket proxy (Go implementation)
create_websocket_proxy() {
    cat > /usr/local/bin/ssh_websocket_proxy.go << 'EOL'
package main

import (
    "log"
    "net"
    "sync"
    "time"

    "github.com/gorilla/websocket"
)

func proxySSH(ws *websocket.Conn, sshConn net.Conn) {
    var wg sync.WaitGroup
    wg.Add(2)

    go func() {
        defer wg.Done()
        for {
            _, data, err := ws.ReadMessage()
            if err != nil {
                break
            }
            sshConn.Write(data)
        }
    }()

    go func() {
        defer wg.Done()
        buffer := make([]byte, 4096)
        for {
            n, err := sshConn.Read(buffer)
            if err != nil {
                break
            }
            ws.WriteMessage(websocket.BinaryMessage, buffer[:n])
        }
    }()

    wg.Wait()
}

func main() {
    upgrader := websocket.Upgrader{
        ReadBufferSize:  4096,
        WriteBufferSize: 4096,
    }

    http.HandleFunc("/ssh", func(w http.ResponseWriter, r *http.Request) {
        ws, err := upgrader.Upgrade(w, r, nil)
        if err != nil {
            log.Println(err)
            return
        }
        defer ws.Close()

        sshConn, err := net.Dial("tcp", "127.0.0.1:22")
        if err != nil {
            log.Println(err)
            return
        }
        defer sshConn.Close()

        proxySSH(ws, sshConn)
    })

    log.Fatal(http.ListenAndServeTLS(":8080", "/etc/ssl/private/server.crt", "/etc/ssl/private/server.key", nil))
}
EOL

    # Compile Go WebSocket proxy
    go build /usr/local/bin/ssh_websocket_proxy.go
}

# Main setup function
main() {
    local domain="$1"

    # Essential packages
    local base_packages=(
        "openssh-server" "fail2ban" "ufw"
        "prometheus" "node-exporter" "certbot"
    )

    install_packages "${base_packages[@]}"
    
    # Generate secure certificates
    generate_certificate "$domain"
    
    # Create optimized WebSocket proxy
    create_websocket_proxy
    
    # Configure SSH with enhanced security
    configure_ssh
    
    # Set up monitoring and logging
    setup_monitoring
}

# Execute main function
main "$@"
