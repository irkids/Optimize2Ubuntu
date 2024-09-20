#!/bin/bash
set -e

# Add Ubuntu 20.04 repository
sudo sh -c 'echo "deb http://archive.ubuntu.com/ubuntu focal main universe" >> /etc/apt/sources.list'

# Update the package list
sudo apt-get update

# Upgrade installed packages
sudo apt-get dist-upgrade -y

# Change DNS server to Google DNS
sudo sed -i 's/nameserver.*/nameserver 8.8.8.8\nnameserver 8.8.4.4/' /etc/resolv.conf
sudo resolvconf -u

# Install necessary packages for productivity and optimization
sudo apt-get install -y jq socat nano htop nload iftop iotop glances nethogs vnstat mtr-tiny nmap tcpdump wireshark netcat iperf3 fping traceroute dnsutils

sudo apt-get install -y apache2 php libapache2-mod-php php-mysql php-cli php-gd php-mbstring php-xml php-zip php-intl php-bcmath php-soap php-curl php-imagick php-redis php-memcached

sudo apt-get install -y redis-server memcached postgresql postgresql-contrib postgresql-client postgresql-server-dev-all

sudo apt-get install -y elasticsearch

sudo apt-get install -y rabbitmq-server

sudo apt-get install -y docker.io docker-compose

sudo apt-get install -y ansible

sudo apt-get install -y awscli google-cloud-sdk

# Update the package list again before installing kubectl
sudo apt-get update

# Install kubectl
sudo apt-get install -y kubectl

sudo apt-get install -y qemu-kvm libvirt-bin virt-manager virt-viewer qemu-system libosinfo-bin

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
