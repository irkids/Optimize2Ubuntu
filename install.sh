#!/bin/bash
set -e

# Add Ubuntu 20.04 repository
sudo sh -c 'echo "deb http://archive.ubuntu.com/ubuntu focal main universe" >> /etc/apt/sources.list'

# Update the package list
sudo apt update

# Upgrade installed packages
sudo apt-get dist-upgrade -y

# Install necessary packages for productivity and optimization
# Group 1: Networking and monitoring tools
sudo apt-get install -y jq socat nano htop nload iftop iotop glances nethogs vnstat mtr-tiny nmap tcpdump wireshark netcat iperf3 fping traceroute dnsutils

# Group 2: Web server and PHP
sudo apt-get install -y apache2 php libapache2-mod-php php-mysql php-cli php-gd php-mbstring php-xml php-zip php-intl php-bcmath php-soap php-curl php-imagick php-redis php-memcached

# Group 3: Database servers
sudo apt-get install -y redis-server memcached postgresql postgresql-contrib postgresql-client postgresql-server-dev-all

# Group 4: Search and analytics engines
sudo apt-get install -y elasticsearch

# Group 5: Messaging and queueing
sudo apt-get install -y rabbitmq-server

# Group 6: Containerization and orchestration
sudo apt-get install -y docker.io docker-compose

# Group 7: Configuration management and provisioning
sudo apt-get install -y ansible

# Group 8: Cloud services
sudo apt-get install -y awscli google-cloud-sdk

# Group 9: Kubernetes
sudo apt-get install -y kubectl

# Group 10: Virtualization and automation
sudo apt-get install -y qemu-kvm libvirt-bin virt-manager virt-viewer qemu-system libosinfo-bin

# Install additional tools for connection quality and speed
sudo apt-get install -y speedtest-cli

# Remove Ubuntu 20.04 repository
sudo sed -i '/focal/d' /etc/apt/sources.list

# Update the package list again
sudo apt update

# Confirm or cancel the reboot
while true; do
    read -p "Do you want to reboot the server? (y/n): " yn
    case $yn in
        [Yy]* ) sudo reboot; break;;
        [Nn]* ) echo "Reboot aborted."; exit;;
        * ) echo "Please answer yes or no.";;
    esac
done
