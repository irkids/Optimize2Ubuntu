#!/bin/bash

# install.sh
# Install script for IRSSH Panel with IKEv2 integration

# Configuration
GITHUB_REPO="https://github.com/irkids/Optimize2Ubuntu/ikev2-script.git"
PANEL_DIR="/opt/irssh-panel"
LOG_FILE="install.log"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a $LOG_FILE
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a $LOG_FILE
    exit 1
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    error "This script must be run as root"
fi

# Install system dependencies
log "Installing system dependencies..."
apt-get update || error "Failed to update package lists"
apt-get install -y git nodejs npm strongswan || error "Failed to install dependencies"

# Create installation directory
log "Creating installation directory..."
mkdir -p $PANEL_DIR
cd $PANEL_DIR || error "Failed to create installation directory"

# Clone IKEv2 script from GitHub
log "Cloning IKEv2 script..."
git clone $GITHUB_REPO scripts/ikev2 || error "Failed to clone IKEv2 script"

# Install React project and dependencies
log "Creating React project..."
npx create-react-app . --template typescript || error "Failed to create React project"

# Install additional dependencies
log "Installing additional dependencies..."
npm install \
    @headlessui/react \
    @heroicons/react \
    tailwindcss \
    postcss \
    autoprefixer \
    recharts \
    axios \
    react-router-dom || error "Failed to install Node dependencies"

# Initialize Tailwind CSS
log "Initializing Tailwind CSS..."
npx tailwindcss init -p

# Create project structure
log "Creating project structure..."
mkdir -p src/{components/{layout,dashboard,users},utils,pages}

# Download and setup components
log "Setting up components..."
cat > src/utils/ikev2.ts << 'EOL'
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);
const SCRIPT_PATH = '/opt/irssh-panel/scripts/ikev2/ikev2.sh';

export class IKEv2Manager {
    async addUser(username: string, password: string) {
        try {
            await execAsync(`sudo ${SCRIPT_PATH} add-user ${username} ${password}`);
            return true;
        } catch (error) {
            console.error('Failed to add user:', error);
            return false;
        }
    }

    async deleteUser(username: string) {
        try {
            await execAsync(`sudo ${SCRIPT_PATH} remove-user ${username}`);
            return true;
        } catch (error) {
            console.error('Failed to delete user:', error);
            return false;
        }
    }

    async listUsers() {
        try {
            const { stdout } = await execAsync(`sudo ${SCRIPT_PATH} list-users`);
            return stdout.trim().split('\n');
        } catch (error) {
            console.error('Failed to list users:', error);
            return [];
        }
    }
}

export default new IKEv2Manager();
EOL

# Setup environment variables
log "Setting up environment variables..."
cat > .env << EOL
REACT_APP_API_URL=http://localhost:3000
REACT_APP_IKEV2_SCRIPT_PATH=/opt/irssh-panel/scripts/ikev2/ikev2.sh
EOL

# Set permissions
log "Setting permissions..."
chown -R $SUDO_USER:$SUDO_USER $PANEL_DIR
chmod +x scripts/ikev2/ikev2.sh

# Create systemd service
log "Creating systemd service..."
cat > /etc/systemd/system/irssh-panel.service << EOL
[Unit]
Description=IRSSH Panel
After=network.target

[Service]
Type=simple
User=$SUDO_USER
WorkingDirectory=$PANEL_DIR
ExecStart=/usr/bin/npm start
Restart=always

[Install]
WantedBy=multi-user.target
EOL

# Start services
log "Starting services..."
systemctl daemon-reload
systemctl enable irssh-panel
systemctl start irssh-panel

# Final setup steps
log "Performing final setup..."
cd $PANEL_DIR
npm run build

# Print success message
log "Installation completed successfully!"
echo -e "${GREEN}The IRSSH Panel has been installed successfully!${NC}"
echo "You can access the panel at: http://localhost:3000"
echo "Installation directory: $PANEL_DIR"
echo "Log file: $LOG_FILE"
