#!/bin/bash
set -e

# Check if running on Ubuntu 22.04 or higher
source /etc/os-release
if [[ "$VERSION_ID" != "22.04" ]]; then
  echo "This script is intended for Ubuntu 22.04. Detected version: $VERSION_ID"
  exit 1
fi

# Update package list and upgrade installed packages
sudo apt-get update
sudo apt-get dist-upgrade -y

# Install necessary tools for Ubuntu Server 22.04
sudo apt-get install -y jq socat nano htop nload iftop iotop glances nethogs vnstat mtr-tiny nmap tcpdump wireshark netcat iperf3 fping traceroute bind9-utils resolvconf

# Install web server and PHP
sudo apt-get install -y apache2 php libapache2-mod-php php-mysql php-cli php-gd php-mbstring php-xml php-zip php-intl php-bcmath php-soap php-curl php-imagick php-redis memcached

# Install database servers
sudo apt-get install -y redis-server postgresql postgresql-contrib postgresql-client

# Install Docker and other container tools
sudo apt-get install -y docker.io docker-compose

# Install Ansible and cloud tools
sudo apt-get install -y ansible awscli

# Install Kubernetes tools
sudo apt-get install -y kubectl

# Install virtualization tools (libvirt-bin is now part of libvirt-daemon-system)
sudo apt-get install -y qemu-kvm virt-manager virt-viewer qemu-system libvirt-daemon-system libvirt-clients

# Install additional tools for connection quality and speed
sudo apt-get install -y speedtest-cli

# Update DNS server to use Google DNS
sudo sed -i 's/nameserver.*/nameserver 8.8.8.8\nnameserver 8.8.4.4/' /etc/resolv.conf

# Confirm or cancel the reboot
while true; do
    read -p "Do you want to reboot the server? (y/n): " yn
    case $yn in
        [Yy]* ) sudo reboot; break;;
        [Nn]* ) echo "Reboot aborted."; exit;;
        * ) echo "Please answer yes or no.";;
    esac
done
