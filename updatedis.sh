#!/bin/bash

# Step 1: Ensure proper permissions and ownership for the web directory
sudo chown -R www-data:www-data /var/www/html
sudo chmod -R 755 /var/www/html

# Step 2: Install required packages if not installed
install_if_missing() {
    if ! dpkg -s "$1" >/dev/null 2>&1; then
        echo "Installing $1..."
        sudo apt-get install -y "$1"
    else
        echo "$1 is already installed."
    fi
}

install_if_missing "php"
install_if_missing "curl"
install_if_missing "nethogs"
install_if_missing "composer"
install_if_missing "npm"

# Step 3: Check for Node.js and install version 16 if necessary
if ! node -v | grep -q "v16"; then
    echo "Installing Node.js version 16..."
    curl -sL https://deb.nodesource.com/setup_16.x | sudo -E bash -
    sudo apt-get install -y nodejs
else
    echo "Node.js version 16 is already installed."
fi

# Step 4: Set up Laravel (if not already set up)
if [ ! -d "/var/www/html/public" ]; then
    echo "Setting up Laravel..."
    cd /var/www/html
    composer create-project --prefer-dist laravel/laravel .
    npm install
else
    echo "Laravel is already set up."
fi

# Build the frontend assets
cd /var/www/html
npm run build

# Step 5: Create the network stats script for SSH users
cat > /var/www/html/app/Scripts/get_network_stats.php << 'EOF'
<?php
function get_network_stats($interface = "eth0") {
    $rx_initial = file_get_contents("/sys/class/net/$interface/statistics/rx_bytes");
    $tx_initial = file_get_contents("/sys/class/net/$interface/statistics/tx_bytes");

    sleep(1);

    $rx_final = file_get_contents("/sys/class/net/$interface/statistics/rx_bytes");
    $tx_final = file_get_contents("/sys/class/net/$interface/statistics/tx_bytes");

    $rx_speed = ($rx_final - $rx_initial);  // Receive speed in bytes
    $tx_speed = ($tx_final - $tx_initial);  // Transmit speed in bytes

    return [
        'rx_speed' => format_speed($rx_speed),
        'tx_speed' => format_speed($tx_speed),
        'rx_total' => format_speed($rx_final),
        'tx_total' => format_speed($tx_final)
    ];
}

function format_speed($bytes) {
    $units = ['B', 'KB', 'MB', 'GB'];
    $pow = $bytes > 0 ? floor(log($bytes) / log(1024)) : 0;
    $bytes /= pow(1024, $pow);
    return round($bytes, 2) . ' ' . $units[$pow] . '/s';
}

header('Content-Type: application/json');
echo json_encode(get_network_stats());
EOF

# Step 6: Add API route for Laravel
cat >> /var/www/html/routes/api.php << 'EOF'
Route::get('/network-stats', function () {
    return response()->json(shell_exec('php /var/www/html/app/Scripts/get_network_stats.php'));
});
EOF

# Step 7: Create the React component to display network stats for each SSH user
cat > /var/www/html/resources/js/components/NetworkStats.js << 'EOF'
import React, { useState, useEffect } from 'react';

const NetworkStats = ({ username }) => {
    const [stats, setStats] = useState({
        rx_speed: '0 B/s',
        tx_speed: '0 B/s',
        rx_total: '0 B',
        tx_total: '0 B'
    });

    useEffect(() => {
        const fetchStats = async () => {
            const response = await fetch('/api/network-stats');
            const data = await response.json();
            setStats(data);
        };

        fetchStats();
        const interval = setInterval(fetchStats, 5000); // Fetch every 5 seconds
        return () => clearInterval(interval);
    }, []);

    return (
        <div>
            <h3>Network Stats for {username}</h3>
            <p>Receive Speed: {stats.rx_speed}</p>
            <p>Transmit Speed: {stats.tx_speed}</p>
            <p>Total Data Received: {stats.rx_total}</p>
            <p>Total Data Sent: {stats.tx_total}</p>
        </div>
    );
};

export default NetworkStats;
EOF

# Step 8: Integrate the NetworkStats component into the dashboard
cat > /var/www/html/resources/js/components/Dashboard.js << 'EOF'
import React from 'react';
import NetworkStats from './NetworkStats';

const Dashboard = () => {
    const onlineUsers = ["user1", "user2"]; // Replace with actual users

    return (
        <div>
            {onlineUsers.map((user) => (
                <div key={user}>
                    <h2>{user}</h2>
                    <NetworkStats username={user} />
                </div>
            ))}
        </div>
    );
};

export default Dashboard;
EOF

# Step 9: Build the frontend assets again after adding the new component
npm run build

echo "Complete setup done with network stats for SSH users!"
