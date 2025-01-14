#!/bin/bash

# Configuration
PANEL_DIR="/opt/irssh-panel"
SCRIPT_URL="https://raw.githubusercontent.com/irkids/Optimize2Ubuntu/refs/heads/main/ikev2-script.py"
LOG_FILE="/root/install.log"
NODE_VERSION="20.10.0"

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

# Create temporary directory for React project
TEMP_DIR=$(mktemp -d)
log "Creating temporary React project..."
cd "$TEMP_DIR" || error "Failed to change to temp directory"
yarn create react-app irssh-panel --template typescript || error "Failed to create React project"

# Create panel directory
log "Creating installation directory..."
mkdir -p "$PANEL_DIR/scripts"

# Download IKEv2 script
log "Downloading IKEv2 script..."
curl -o "$PANEL_DIR/scripts/ikev2.py" "$SCRIPT_URL" || error "Failed to download IKEv2 script"
chmod +x "$PANEL_DIR/scripts/ikev2.py"
sed -i 's/\r$//' "$PANEL_DIR/scripts/ikev2.py"

# Move React project files
log "Moving React project files..."
mv "$TEMP_DIR/irssh-panel/"* "$TEMP_DIR/irssh-panel/".* "$PANEL_DIR/" 2>/dev/null || true
cd "$PANEL_DIR" || error "Failed to change directory"
rm -rf "$TEMP_DIR"

# Install dependencies
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
cat > tailwind.config.js << 'EOF'
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

# Create IKEv2Users component
log "Creating IKEv2Users component..."
cat > src/components/users/IKEv2Users.tsx << 'EOL'
import React, { useState, useEffect } from 'react';
import { PlusIcon, TrashIcon } from '@heroicons/react/24/outline';
import IKEv2Manager from '../../utils/ikev2';

const IKEv2Users: React.FC = () => {
  const [users, setUsers] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [newUsername, setNewUsername] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [isAddingUser, setIsAddingUser] = useState(false);

  useEffect(() => {
    fetchUsers();
  }, []);

  const fetchUsers = async () => {
    try {
      setIsLoading(true);
      const userList = await IKEv2Manager.listUsers();
      setUsers(userList);
      setError(null);
    } catch (err) {
      setError('Failed to fetch users');
    } finally {
      setIsLoading(false);
    }
  };

  const handleAddUser = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newUsername || !newPassword) return;

    try {
      setIsAddingUser(true);
      const success = await IKEv2Manager.addUser(newUsername, newPassword);
      if (success) {
        await fetchUsers();
        setNewUsername('');
        setNewPassword('');
        setIsAddingUser(false);
      } else {
        setError('Failed to add user');
      }
    } catch (err) {
      setError('Failed to add user');
    } finally {
      setIsAddingUser(false);
    }
  };

  const handleDeleteUser = async (username: string) => {
    try {
      const success = await IKEv2Manager.deleteUser(username);
      if (success) {
        await fetchUsers();
      } else {
        setError('Failed to delete user');
      }
    } catch (err) {
      setError('Failed to delete user');
    }
  };

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-6">IKEv2 Users Management</h2>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      <form onSubmit={handleAddUser} className="mb-8 bg-white p-4 rounded shadow">
        <div className="flex flex-col md:flex-row gap-4">
          <input
            type="text"
            value={newUsername}
            onChange={(e) => setNewUsername(e.target.value)}
            placeholder="Username"
            className="flex-1 px-4 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <input
            type="password"
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
            placeholder="Password"
            className="flex-1 px-4 py-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            type="submit"
            disabled={isAddingUser}
            className="bg-blue-500 text-white px-6 py-2 rounded hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 flex items-center justify-center"
          >
            <PlusIcon className="h-5 w-5 mr-2" />
            {isAddingUser ? 'Adding...' : 'Add User'}
          </button>
        </div>
      </form>

      <div className="bg-white rounded shadow overflow-hidden">
        <div className="border-b px-6 py-4 font-semibold text-gray-700">
          Active Users
        </div>
        <div className="divide-y">
          {isLoading ? (
            <div className="px-6 py-4 text-gray-500">Loading users...</div>
          ) : users.length === 0 ? (
            <div className="px-6 py-4 text-gray-500">No users found</div>
          ) : (
            users.map((user) => (
              <div key={user} className="px-6 py-4 flex items-center justify-between">
                <span className="text-gray-700">{user}</span>
                <button
                  onClick={() => handleDeleteUser(user)}
                  className="text-red-500 hover:text-red-600 p-2 rounded focus:outline-none focus:ring-2 focus:ring-red-500"
                >
                  <TrashIcon className="h-5 w-5" />
                </button>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default IKEv2Users;
EOL

# Create Layout component
log "Creating Layout component..."
cat > src/components/layout/MainLayout.tsx << 'EOL'
import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { HomeIcon, UserGroupIcon } from '@heroicons/react/24/outline';

const MainLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const location = useLocation();

  const navigation = [
    { name: 'Dashboard', href: '/', icon: HomeIcon },
    { name: 'IKEv2 Users', href: '/ikev2-users', icon: UserGroupIcon },
  ];

  return (
    <div className="min-h-screen bg-gray-100">
      <div className="flex">
        {/* Sidebar */}
        <div className="w-64 bg-white h-screen shadow-lg">
          <div className="px-6 py-4 border-b">
            <h1 className="text-xl font-bold">IRSSH Panel</h1>
          </div>
          <nav className="mt-4">
            {navigation.map((item) => {
              const isActive = location.pathname === item.href;
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={`flex items-center px-6 py-3 text-sm font-medium ${
                    isActive
                      ? 'text-blue-600 bg-blue-50'
                      : 'text-gray-600 hover:text-blue-600 hover:bg-blue-50'
                  }`}
                >
                  <item.icon className="h-5 w-5 mr-3" />
                  {item.name}
                </Link>
              );
            })}
          </nav>
        </div>

        {/* Main content */}
        <div className="flex-1">
          <main className="p-6">{children}</main>
        </div>
      </div>
    </div>
  );
};

export default MainLayout;
EOL

# Create Dashboard component
log "Creating Dashboard component..."
cat > src/components/dashboard/Dashboard.tsx << 'EOL'
import React from 'react';
import { ArrowTrendingUpIcon, UserGroupIcon } from '@heroicons/react/24/outline';

const Dashboard: React.FC = () => {
  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-6">Dashboard</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Active Users Card */}
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <UserGroupIcon className="h-8 w-8 text-blue-500" />
            <div className="ml-4">
              <h3 className="text-sm font-medium text-gray-500">Active Users</h3>
              <p className="text-2xl font-semibold">0</p>
            </div>
          </div>
        </div>

        {/* Network Traffic Card */}
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center">
            <ArrowTrendingUpIcon className="h-8 w-8 text-green-500" />
            <div className="ml-4">
              <h3 className="text-sm font-medium text-gray-500">Network Traffic</h3>
              <p className="text-2xl font-semibold">0 MB/s</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
EOL

# Update App.tsx with routing
log "Setting up routing..."
cat > src/App.tsx << 'EOL'
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import MainLayout from './components/layout/MainLayout';
import Dashboard from './components/dashboard/Dashboard';
import IKEv2Users from './components/users/IKEv2Users';
import './App.css';

function App() {
  return (
    <Router>
      <MainLayout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/ikev2-users" element={<IKEv2Users />} />
        </Routes>
      </MainLayout>
    </Router>
  );
}

export default App;
EOL

# Update index.css with Tailwind
log "Setting up Tailwind CSS..."
cat > src/index.css << 'EOL'
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  margin: 0;
  font-family: -apple-system, system-ui, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
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

# Create REST API server for IKEv2
log "Setting up API server..."
cat > server.js << \EOF

#!/usr/bin/env node
const express = require('express');
const cors = require('cors');
const { exec } = require('child_process');
const { promisify } = require('util');
const execAsync = promisify(exec);

const app = express();
app.use(cors({
    origin: 'http://localhost:3001',
    methods: ['GET', 'POST', 'DELETE'],
    credentials: true
}));
app.use(express.json());

const SCRIPT_PATH = '/opt/irssh-panel/scripts/ikev2.py';

// Middleware for error handling
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ error: 'Internal server error' });
});

// Middleware for logging
app.use((req, res, next) => {
    console.log(`${new Date().toISOString()} ${req.method} ${req.url}`);
    next();
});

app.get('/api/ikev2/users', async (req, res) => {
    try {
        console.log('Executing list-users command...');
        const { stdout } = await execAsync(`sudo ${SCRIPT_PATH} list-users`);
        const users = stdout.trim().split('\n').filter(Boolean);
        console.log('Users found:', users);
        res.json(users);
    } catch (error) {
        console.error('Error listing users:', error);
        res.status(500).json({ error: 'Failed to list users', details: error.message });
    }
});

app.post('/api/ikev2/users', async (req, res) => {
    const { username, password } = req.body;
    if (!username || !password) {
        return res.status(400).json({ error: 'Username and password are required' });
    }

    try {
        console.log(`Adding user: ${username}`);
        await execAsync(`sudo ${SCRIPT_PATH} add-user ${username} ${password}`);
        console.log(`Successfully added user: ${username}`);
        res.json({ success: true });
    } catch (error) {
        console.error('Error adding user:', error);
        res.status(500).json({ error: 'Failed to add user', details: error.message });
    }
});

app.delete('/api/ikev2/users/:username', async (req, res) => {
    const { username } = req.params;
    try {
        console.log(`Removing user: ${username}`);
        await execAsync(`sudo ${SCRIPT_PATH} remove-user ${username}`);
        console.log(`Successfully removed user: ${username}`);
        res.json({ success: true });
    } catch (error) {
        console.error('Error removing user:', error);
        res.status(500).json({ error: 'Failed to remove user', details: error.message });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
EOL
const express = require('express');
const cors = require('cors');
const { exec } = require('child_process');
const { promisify } = require('util');
const execAsync = promisify(exec);

const app = express();
app.use(cors());
app.use(express.json());

const SCRIPT_PATH = '/opt/irssh-panel/scripts/ikev2.py';

app.get('/api/ikev2/users', async (req, res) => {
  try {
    const { stdout } = await execAsync(`sudo python3 ${SCRIPT_PATH} list-users`);
    const users = stdout.trim().split('\n').filter(Boolean);
    res.json(users);
  } catch (error) {
    console.error('Error listing users:', error);
    res.status(500).json({ error: 'Failed to list users' });
  }
});

app.post('/api/ikev2/users', async (req, res) => {
  const { username, password } = req.body;
  if (!username || !password) {
    return res.status(400).json({ error: 'Username and password are required' });
  }

  try {
    await execAsync(`sudo python3 ${SCRIPT_PATH} add-user ${username} ${password}`);
    res.json({ success: true });
  } catch (error) {
    console.error('Error adding user:', error);
    res.status(500).json({ error: 'Failed to add user' });
  }
});

app.delete('/api/ikev2/users/:username', async (req, res) => {
  const { username } = req.params;
  try {
    await execAsync(`sudo python3 ${SCRIPT_PATH} remove-user ${username}`);
    res.json({ success: true });
  } catch (error) {
    console.error('Error removing user:', error);
    res.status(500).json({ error: 'Failed to remove user' });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
EOL

# Install API server dependencies
log "Installing API server dependencies..."
yarn add express cors

# Set up sudo access for API server
log "Setting up sudo access..."
echo "irssh ALL=(ALL) NOPASSWD: /opt/irssh-panel/scripts/ikev2.py" > /etc/sudoers.d/irssh-panel
chmod 0440 /etc/sudoers.d/irssh-panel

# Create API user
log "Creating API user..."
useradd -r -s /bin/false irssh
chown -R irssh:irssh "$PANEL_DIR"
chmod 755 "$PANEL_DIR/scripts/ikev2.py"

# Create service for API server
log "Creating API server service..."
cat > /etc/systemd/system/irssh-api.service << EOL
[Unit]
Description=IRSSH API Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$PANEL_DIR
ExecStart=/usr/bin/node server.js
Restart=always
Environment=PORT=3000

[Install]
WantedBy=multi-user.target
EOL

# Create service for React app
log "Creating React app service..."
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
Environment=PORT=3001

[Install]
WantedBy=multi-user.target
EOL

# Update React app port in environment
sed -i 's/REACT_APP_API_URL=http:\/\/localhost:3000/REACT_APP_API_URL=http:\/\/localhost:3000/g' .env

# Start services
log "Starting services..."
systemctl daemon-reload
systemctl enable irssh-api
systemctl enable irssh-panel
systemctl start irssh-api
systemctl start irssh-panel

# Clean up
log "Cleaning up..."
rm -rf "$PANEL_DIR/irssh-panel"

# Success message
log "Installation completed successfully!"
echo -e "${GREEN}IRSSH Panel has been installed!${NC}"
echo "Access the panel at: http://localhost:3001"
echo "API server running at: http://localhost:3000"
echo "Installation directory: $PANEL_DIR"
echo "Log file: $LOG_FILE"
