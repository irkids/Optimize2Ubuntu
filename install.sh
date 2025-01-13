#!/bin/bash

# Configuration
PANEL_DIR="/opt/irssh-panel"
SCRIPT_URL="https://raw.githubusercontent.com/irkids/Optimize2Ubuntu/refs/heads/main/ikev2-script.py"
LOG_FILE="/root/install.log"

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

# Clean previous installation
log "Cleaning previous installation..."
if [ -d "$PANEL_DIR" ]; then
    systemctl stop irssh-panel >/dev/null 2>&1
    systemctl disable irssh-panel >/dev/null 2>&1
    rm -rf "$PANEL_DIR"
fi

# Create fresh directory
log "Creating installation directory..."
mkdir -p "$PANEL_DIR/scripts"
cd "$PANEL_DIR" || error "Failed to change directory"

# Download IKEv2 script
log "Downloading IKEv2 script..."
curl -o scripts/ikev2.py "$SCRIPT_URL" || error "Failed to download IKEv2 script"
chmod +x scripts/ikev2.py
sed -i 's/\r$//' scripts/ikev2.py

# Create temporary React project
log "Creating temporary React project..."
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR" || error "Failed to change to temp directory"
npx create-react-app irssh-panel --template typescript || error "Failed to create React project"

# Move React project files
log "Setting up React project..."
mv irssh-panel/* irssh-panel/.* "$PANEL_DIR" 2>/dev/null || true
cd "$PANEL_DIR" || error "Failed to change directory"

# Install dependencies
log "Installing dependencies..."
npm install \
    @headlessui/react \
    @heroicons/react \
    tailwindcss \
    postcss \
    autoprefixer \
    recharts \
    axios \
    react-router-dom || error "Failed to install dependencies"

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
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);
const SCRIPT_PATH = '/opt/irssh-panel/scripts/ikev2.py';

export class IKEv2Manager {
    async addUser(username: string, password: string) {
        try {
            await execAsync(`sudo python3 ${SCRIPT_PATH} add-user ${username} ${password}`);
            return true;
        } catch (error) {
            console.error('Failed to add user:', error);
            return false;
        }
    }

    async deleteUser(username: string) {
        try {
            await execAsync(`sudo python3 ${SCRIPT_PATH} remove-user ${username}`);
            return true;
        } catch (error) {
            console.error('Failed to delete user:', error);
            return false;
        }
    }

    async listUsers() {
        try {
            const { stdout } = await execAsync(`sudo python3 ${SCRIPT_PATH} list-users`);
            return stdout.trim().split('\n');
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
ExecStart=/usr/bin/npm start
Restart=always

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
rm -rf "$TEMP_DIR"

# Success message
log "Installation completed successfully!"
echo -e "${GREEN}IRSSH Panel has been installed!${NC}"
echo "Access the panel at: http://localhost:3000"
echo "Installation directory: $PANEL_DIR"
echo "Log file: $LOG_FILE"
