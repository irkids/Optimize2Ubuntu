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

sudo apt-get update

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
