#!/bin/bash

# Configuration
PANEL_DIR="/opt/irssh-panel"
BACKEND_DIR="$PANEL_DIR/backend"
FRONTEND_DIR="$PANEL_DIR/frontend"
SCRIPT_URL="https://raw.githubusercontent.com/irkids/Optimize2Ubuntu/refs/heads/main/ikev2-script.py"
LOG_FILE="/root/install.log"
GITHUB_REPO="https://raw.githubusercontent.com/your-repo"

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

# Create directories
log "Creating installation directories..."
mkdir -p "$PANEL_DIR"/{frontend,backend,scripts}
cd "$PANEL_DIR" || error "Failed to change directory"

# Install system dependencies
log "Installing system dependencies..."
apt-get update
apt-get install -y python3-pip nodejs npm curl git

# Setup Backend
log "Setting up backend..."
cd "$BACKEND_DIR" || error "Failed to change to backend directory"
cat > requirements.txt << EOL
fastapi==0.68.1
uvicorn==0.15.0
python-jose==3.3.0
passlib==1.7.4
python-multipart==0.0.5
sqlalchemy==1.4.23
pymysql==1.0.2
EOL

pip3 install -r requirements.txt

# Create backend structure
mkdir -p app/{api,core,db,modules,utils}
touch app/{__init__.py,main.py}

# Setup Frontend
log "Setting up frontend..."
cd "$FRONTEND_DIR" || error "Failed to change to frontend directory"

# Initialize React project with TypeScript
npx create-react-app . --template typescript

# Install frontend dependencies
npm install \
    @headlessui/react \
    @heroicons/react \
    tailwindcss \
    postcss \
    autoprefixer \
    recharts \
    axios \
    react-router-dom \
    @types/react-router-dom \
    lucide-react

# Setup Tailwind
npx tailwindcss init -p

# Create frontend structure
mkdir -p src/{components,contexts,hooks,pages,utils}/{layout,dashboard,users,common}

# Download component files
log "Downloading frontend components..."
COMPONENTS=(
    "MainLayout"
    "Dashboard"
    "UserManagement"
    "AuthContext"
    "ThemeContext"
    "GlobalStore"
)

for component in "${COMPONENTS[@]}"; do
    curl -o "src/components/${component}.tsx" "$GITHUB_REPO/components/${component}.tsx" || log "Warning: Failed to download ${component}"
done

# Setup API configuration
log "Configuring API..."
cat > src/utils/api.ts << EOL
import axios from 'axios';

export const api = axios.create({
    baseURL: process.env.REACT_APP_API_URL || 'http://localhost:3000/api',
    headers: {
        'Content-Type': 'application/json',
    },
});

api.interceptors.request.use(
    (config) => {
        const token = localStorage.getItem('token');
        if (token) {
            config.headers.Authorization = \`Bearer \${token}\`;
        }
        return config;
    },
    (error) => Promise.reject(error)
);
EOL

# Create environment files
log "Creating environment files..."
cat > .env << EOL
REACT_APP_API_URL=http://localhost:3000
EOL

# Build frontend
log "Building frontend..."
npm run build

# Setup systemd services
log "Creating systemd services..."

# Backend service
cat > /etc/systemd/system/irssh-backend.service << EOL
[Unit]
Description=IRSSH Backend
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$BACKEND_DIR
ExecStart=/usr/bin/uvicorn app.main:app --host 0.0.0.0 --port 3000
Restart=always

[Install]
WantedBy=multi-user.target
EOL

# Frontend service
cat > /etc/systemd/system/irssh-frontend.service << EOL
[Unit]
Description=IRSSH Frontend
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$FRONTEND_DIR
ExecStart=/usr/bin/npm start
Restart=always

[Install]
WantedBy=multi-user.target
EOL

# Start services
log "Starting services..."
systemctl daemon-reload
systemctl enable irssh-backend irssh-frontend
systemctl start irssh-backend irssh-frontend

# Success message
log "Installation completed successfully!"
echo -e "${GREEN}IRSSH Panel has been installed!${NC}"
echo "Backend API: http://localhost:3000"
echo "Frontend: http://localhost:3001"
echo "Installation directory: $PANEL_DIR"
echo "Log file: $LOG_FILE"
