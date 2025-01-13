#!/bin/bash

# Configuration
PANEL_DIR="/opt/irssh-panel"
SCRIPT_URL="https://raw.githubusercontent.com/irkids/Optimize2Ubuntu/refs/heads/main/ikev2-script.py"
LOG_FILE="/root/install.log"
NODE_VERSION="20.10.0"  # Specify exact Node.js version

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Logging
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a $LOG_FILE
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a $LOG_FILE
    exit 1
}

# Check root
if [[ $EUID -ne 0 ]]; then
    error "This script must be run as root"
fi

# Install Node.js
install_nodejs() {
    log "Installing Node.js $NODE_VERSION..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - || error "Failed to setup Node.js repository"
    apt-get install -y nodejs || error "Failed to install Node.js"
    
    # Install yarn
    log "Installing Yarn..."
    npm install -g yarn || error "Failed to install Yarn"
}

# Clean previous installation
log "Cleaning previous installation..."
if [ -d "$PANEL_DIR" ]; then
    systemctl stop irssh-panel >/dev/null 2>&1
    systemctl disable irssh-panel >/dev/null 2>&1
    rm -rf "$PANEL_DIR"
fi

# Install Node.js
install_nodejs

# Create fresh directory
log "Creating installation directory..."
mkdir -p "$PANEL_DIR/scripts"
cd "$PANEL_DIR" || error "Failed to change directory"

# Download IKEv2 script
log "Downloading IKEv2 script..."
curl -o scripts/ikev2.py "$SCRIPT_URL" || error "Failed to download IKEv2 script"
chmod +x scripts/ikev2.py
sed -i 's/\r$//' scripts/ikev2.py

# Create React project using Yarn
log "Creating React project..."
yarn create react-app irssh-panel --template typescript || error "Failed to create React project"

# Move React project files
log "Setting up React project..."
mv irssh-panel/* irssh-panel/.* "$PANEL_DIR" 2>/dev/null || true
cd "$PANEL_DIR" || error "Failed to change directory"

# Install dependencies with specific versions
log "Installing dependencies..."
yarn add \
    @headlessui/react@1.7.17 \
    @heroicons/react@2.1.1 \
    tailwindcss@3.4.1 \
    postcss@8.4.33 \
    autoprefixer@10.4.16 \
    recharts@2.10.3 \
    axios@1.6.5 \
    react-router-dom@6.21.1 || error "Failed to install dependencies"

# Initialize Tailwind
log "Setting up Tailwind CSS..."
cat > tailwind.config.js << 'EOL'
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: { extend: {} },
  plugins: [],
}
EOL

cat > postcss.config.js << 'EOL'
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
EOL

# Setup project structure
log "Creating project structure..."
mkdir -p src/{components/{layout,dashboard,users},utils,pages}

# Create IKEv2 utility
log "Setting up IKEv2 integration..."
cat > src/utils/ikev2.ts << 'EOL'
import axios from 'axios';

export class IKEv2Manager {
    private baseUrl: string;

    constructor() {
        this.baseUrl = process.env.REACT_APP_API_URL || 'http://localhost:3000';
    }

    async addUser(username: string, password: string): Promise<boolean> {
        try {
            const response = await axios.post(`${this.baseUrl}/api/ikev2/users`, { 
                username, 
                password 
            });
            return response.status === 200;
        } catch (error) {
            console.error('Failed to add user:', error);
            return false;
        }
    }

    async deleteUser(username: string): Promise<boolean> {
        try {
            const response = await axios.delete(`${this.baseUrl}/api/ikev2/users/${username}`);
            return response.status === 200;
        } catch (error) {
            console.error('Failed to delete user:', error);
            return false;
        }
    }

    async listUsers(): Promise<string[]> {
        try {
            const response = await axios.get(`${this.baseUrl}/api/ikev2/users`);
            return response.data;
        } catch (error) {
            console.error('Failed to list users:', error);
            return [];
        }
    }
}

export default new IKEv2Manager();
EOL

# Setup environment
log "Configuring environment..."
cat > .env << EOL
REACT_APP_API_URL=http://localhost:3000
REACT_APP_IKEV2_SCRIPT_PATH=/opt/irssh-panel/scripts/ikev2.py
EOL

# Set permissions
log "Setting permissions..."
chown -R root:root "$PANEL_DIR"
chmod -R 755 "$PANEL_DIR"

# Create service
log "Creating systemd service..."
cat > /etc/systemd/system/irssh-panel.service << EOL
[Unit]
Description=IRSSH Panel
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$PANEL_DIR
ExecStart=/usr/bin/yarn start
Restart=always
Environment=PORT=3000

[Install]
WantedBy=multi-user.target
EOL

# Start service
log "Starting service..."
systemctl daemon-reload
systemctl enable irssh-panel
systemctl start irssh-panel

# Clean up
log "Cleaning up..."
rm -rf "$PANEL_DIR/irssh-panel"

# Success message
log "Installation completed successfully!"
echo -e "${GREEN}IRSSH Panel has been installed!${NC}"
echo "Access the panel at: http://localhost:3000"
echo "Installation directory: $PANEL_DIR"
echo "Log file: $LOG_FILE"
