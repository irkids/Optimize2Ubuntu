#!/bin/bash

# Ensure proper permissions and ownership
sudo chown -R www-data:www-data /var/www/html
sudo chmod -R 755 /var/www/html

# Step 1: Clean the existing directory or backup
if [ -d "/var/www/html" ]; then
    echo "Backing up and cleaning existing /var/www/html directory..."
    sudo mv /var/www/html /var/www/html_backup
    sudo mkdir /var/www/html
fi

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

# Step 3: Install Node.js (Version 16)
if ! node -v | grep -q "v16"; then
    echo "Installing Node.js version 16..."
    curl -sL https://deb.nodesource.com/setup_16.x | sudo -E bash -
    sudo apt-get install -y nodejs
else
    echo "Node.js version 16 is already installed."
fi

# Step 4: Set up Laravel project
cd /var/www/html
composer create-project --prefer-dist laravel/laravel .
npm install

# Step 5: Build the frontend
npm run build

# Step 6: Add network stats script
cat > /var/www/html/app/Scripts/get_network_stats.php << 'EOF'
<?php
function get_network_stats($interface = "eth0") {
    $rx_initial = file_get_contents("/sys/class/net/$interface/statistics/rx_bytes");
    $tx_initial = file_get_contents("/sys/class/net/$interface/statistics/tx_bytes");

    sleep(1);

    $rx_final = file_get_contents("/sys/class/net/$interface/statistics/rx_bytes");
    $tx_final = file_get_contents("/sys/class/net/$interface/statistics/tx_bytes");

    $rx_speed = ($rx_final - $rx_initial);
    $tx_speed = ($tx_final - $tx_initial);

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

# Step 7: Add API route
cat >> /var/www/html/routes/api.php << 'EOF'
Route::get('/network-stats', function () {
    return response()->json(shell_exec('php /var/www/html/app/Scripts/get_network_stats.php'));
});
EOF

# Step 8: Add React component
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
        const interval = setInterval(fetchStats, 5000);
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

# Step 9: Build the frontend again
npm run build

echo "Setup complete with network stats for SSH users!"
