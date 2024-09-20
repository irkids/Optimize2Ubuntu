#!/bin/bash

# Change the DNS of the server to Google DNS
sudo cp /etc/resolv.conf /etc/resolv.conf.backup
echo "nameserver 8.8.8.8" | sudo tee -a /etc/resolv.conf
echo "nameserver 8.8.4.4" | sudo tee -a /etc/resolv.conf

# Update the package list
sudo apt-get update

# Upgrade installed packages
sudo apt-get dist-upgrade -y

# Install necessary packages for productivity and optimization
sudo apt-get install -y \
    jq \                  # A lightweight and flexible command-line JSON processor
    pkg-config \          # A helper tool for building and installing libraries
    libssl-dev \          # Development files for OpenSSL
    curl \                # A tool for transferring data using URLs
    socat \               # A relay for bidirectional data transfer between sockets
    nano \               # A powerful text editor
    htop \                # An interactive process viewer
    nload \               # A network traffic monitor
    iftop \               # A network traffic analyzer
    iotop \               # An I/O traffic monitor
    glances \             # A system monitoring tool
    nethogs \             # A network traffic monitor
    vnstat \              # A network traffic monitor
    mtr \                 # A network diagnostic tool
    nmap \                # A network scanner
    tcpdump \             # A network packet analyzer
    wireshark \           # A network protocol analyzer
    netcat \              # A network debugging tool
    iperf3 \              # A network performance measurement tool
    fping \               # A network latency measurement tool
    traceroute \          # A network tracing tool
    dnsutils \            # DNS utilities
    apache2 \             # The Apache web server
    php \                 # The PHP programming language
    libapache2-mod-php \  # Apache module for PHP
    php-mysql \           # PHP MySQL module
    php-cli \             # PHP command-line interpreter
    php-gd \              # PHP GD library
    php-mbstring \        # PHP multibyte string support
    php-xml \             # PHP XML support
    php-zip \             # PHP ZIP support
    php-intl \            # PHP internationalization support
    php-bcmath \          # PHP arbitrary precision mathematics support
    php-soap \            # PHP SOAP support
    php-curl \            # PHP cURL support
    php-imagick \         # PHP Imagick support
    php-redis \           # PHP Redis support
    php-memcached \       # PHP Memcached support
    php-xdebug \          # PHP Xdebug support
    php-apcu \            # PHP APCu support
    php-pear \            # PHP PEAR package manager
    php-dev \             # PHP development files
    php-fpm \             # PHP FastCGI Process Manager
    nodejs \              # The Node.js JavaScript runtime
    npm \                 # The Node.js package manager
    redis-server \        # Redis in-memory data structure store
    memcached \           # Memcached distributed memory object caching system
    mongodb \             # MongoDB NoSQL document-oriented database
    postgresql \          # PostgreSQL relational database system
    postgresql-contrib \  # PostgreSQL contrib modules
    postgresql-client \   # PostgreSQL client
    postgresql-server-dev-all \ # PostgreSQL server development files
    elasticsearch \       # Elasticsearch search and analytics engine
    rabbitmq-server \     # RabbitMQ message broker
    zookeeper \           # Apache ZooKeeper distributed coordination service
    docker.io \           # Docker container platform
    docker-compose \      # Docker Compose
    ansible \             # Ansible, a configuration management and provisioning tool
    awscli \              # AWS Command Line Interface
    google-cloud-sdk \    # Google Cloud SDK
    kubectl \             # Kubernetes command-line tool
    minikube \            # Minikube local Kubernetes cluster manager
    virtualbox \          # Oracle VirtualBox virtualization software
    vagrant \             # Vagrant virtual machine automation tool
    libvirt-bin \         # libvirt virtualization API
    qemu-kvm \            # QEMU virtualization software
    libvirt-daemon-system \ # Enable libvirt daemon at boot
    libvirt-clients \     # libvirt clients
    virt-manager \        # Virtual Machine Manager GUI
    virt-viewer \         # Virtual Machine Viewer GUI
    qemu-system \         # QEMU virtualization system
    libosinfo-bin \       # libosinfo library
    libvirt-dev \         # libvirt development files
    libvirt-doc \         # libvirt documentation
    libvirt-utils \       # libvirt utilities
    libvirt-python \      # libvirt Python bindings
    libvirt-python3 \     # libvirt Python 3 bindings
    python-libvirt \      # Python libvirt library
    python3-libvirt \     # Python 3 libvirt library

# Confirm or cancel the reboot
while true; do
    read -p "Do you want to reboot the server? (y/n): " yn
    case $yn in
        [Yy]* ) sudo reboot; break;;
        [Nn]* ) echo "Reboot aborted."; exit;;
        * ) echo "Please answer yes or no.";;
    esac
done
