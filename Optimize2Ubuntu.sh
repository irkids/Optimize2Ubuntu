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
    redis-server \        # The Redis in-memory data structure store
    memcached \           # The Memcached distributed memory object caching system
    mongodb \             # The MongoDB NoSQL document-oriented database
    postgresql \          # The PostgreSQL relational database system
    postgresql-contrib \  # PostgreSQL contrib modules
    postgresql-client \   # PostgreSQL client
    postgresql-server-dev-all \ # PostgreSQL server development files
    elasticsearch \       # The Elasticsearch search and analytics engine
    rabbitmq-server \     # The RabbitMQ message broker
    zookeeper \           # The Apache ZooKeeper distributed coordination service
    docker.io \           # The Docker container platform
    docker-compose \      # Docker Compose
    ansible \             # Ansible, a configuration management and provisioning tool
    awscli \              # The AWS Command Line Interface
    google-cloud-sdk \    # The Google Cloud SDK
    kubectl \             # The Kubernetes command-line tool
    minikube \            # The Minikube local Kubernetes cluster manager
    virtualbox \          # The Oracle VirtualBox virtualization software
    vagrant \             # The Vagrant virtual machine automation tool
    libvirt-bin \         # The libvirt virtualization API
    qemu-kvm \            # The QEMU virtualization software
    libvirt-daemon-system \ # The libvirt virtualization daemon
    libvirt-clients \     # The libvirt virtualization clients
    virt-manager \        # The Virtual Machine Manager GUI
    virt-viewer \         # The Virtual Machine Viewer GUI
    qemu-system \         # The QEMU virtualization system
    libosinfo-bin \       # The libosinfo library
    libvirt-dev \         # The libvirt development files
    libvirt-doc \         # The libvirt documentation
    libvirt-utils \       # The libvirt utilities
    libvirt-python \      # The libvirt Python bindings
    libvirt-python3 \     # The libvirt Python 3 bindings
    python-libvirt \      # The Python libvirt library
    python3-libvirt \     # The Python 3 libvirt library
    libvirt-daemon-system \ # Enable libvirt daemon at boot
    libvirt-clients \     # Install libvirt clients
    virt-manager \        # Install Virtual Machine Manager GUI
    virt-viewer \         # Install Virtual Machine Viewer GUI
    qemu-system \         # Install QEMU virtualization system
    libosinfo-bin \       # Install libosinfo library
    libvirt-dev \         # Install libvirt development files
    libvirt-doc \         # Install libvirt documentation
    libvirt-utils \       # Install libvirt utilities
    libvirt-python \      # Install libvirt Python bindings
    libvirt-python3 \     # Install libvirt Python 3 bindings
    python-libvirt \      # Install Python libvirt library
    python3-libvirt \     # Install Python 3 libvirt library
    nano \               # Install nano text editor (already included)
    htop \                # Install htop process viewer (already included)
    curl \                # Install curl (already included)
    socat \               # Install socat (already included)
    net-tools \           # Install net-tools (already included)
    nload \               # Install nload network traffic monitor
    iftop \               # Install iftop network traffic analyzer
    iotop \               # Install iotop I/O traffic monitor
    glances \             # Install glances system monitoring tool
    nethogs \             # Install nethogs network traffic monitor
    vnstat \              # Install vnstat network traffic monitor
    mtr \                 # Install mtr network diagnostic tool
    nmap \                # Install nmap network scanner
    tcpdump \             # Install tcpdump network packet analyzer
    wireshark \           # Install wireshark network protocol analyzer
    netcat \              # Install netcat network debugging tool
    iperf3 \              # Install iperf3 network performance measurement tool
    fping \               # Install fping network latency measurement tool
    traceroute \          # Install traceroute network tracing tool
    dnsutils \            # Install dnsutils DNS utilities
    apache2 \             # Install Apache web server (already included)
    php \                 # Install PHP programming language (already included)
    libapache2-mod-php \  # Install Apache module for PHP (already included)
    php-mysql \           # Install PHP MySQL module (already included)
    php-cli \             # Install PHP command-line interpreter (already included)
    php-gd \              # Install PHP GD library (already included)
    php-mbstring \        # Install PHP multibyte string support (already included)
    php-xml \             # Install PHP XML support (already included)
    php-zip \             # Install PHP ZIP support (already included)
    php-intl \            # Install PHP internationalization support (already included)
    php-bcmath \          # Install PHP arbitrary precision mathematics support (already included)
    php-soap \            # Install PHP SOAP support (already included)
    php-curl \            # Install PHP cURL support (already included)
    php-imagick \         # Install PHP Imagick support (already included)
    php-redis \           # Install PHP Redis support (already included)
    php-memcached \       # Install PHP Memcached support (already included)
    php-xdebug \          # Install PHP Xdebug support (already included)
    php-apcu \            # Install PHP APCu support (already included)
    php-pear \            # Install PHP PEAR package manager (already included)
    php-dev \             # Install PHP development files (already included)
    php-fpm \             # Install PHP FastCGI Process Manager (already included)
    nodejs \              # Install Node.js JavaScript runtime (already included)
    npm \                 # Install Node.js package manager (already included)
    redis-server \        # Install Redis in-memory data structure store (already included)
    memcached \           # Install Memcached distributed memory object caching system (already included)
    mongodb \             # Install MongoDB NoSQL document-oriented database (already included)
    postgresql \          # Install PostgreSQL relational database system (already included)
    postgresql-contrib \  # Install PostgreSQL contrib modules (already included)
    postgresql-client \   # Install PostgreSQL client (already included)
    postgresql-server-dev-all \ # Install PostgreSQL server development files (already included)
    elasticsearch \       # Install Elasticsearch search and analytics engine (already included)
    rabbitmq-server \     # Install RabbitMQ message broker (already included)
    zookeeper \           # Install Apache ZooKeeper distributed coordination service (already included)
    docker.io \           # Install Docker container platform (already included)
    docker-compose \      # Install Docker Compose (already included)
    ansible \             # Install Ansible, a configuration management and provisioning tool (already included)
    awscli \              # Install AWS Command Line Interface (already included)
    google-cloud-sdk \    # Install Google Cloud SDK (already included)
    kubectl \             # Install Kubernetes command-line tool (already included)
    minikube \            # Install Minikube local Kubernetes cluster manager (already included)
    virtualbox \          # Install Oracle VirtualBox virtualization software (already included)
    vagrant \             # Install Vagrant virtual machine automation tool (already included)
    libvirt-bin \         # Install libvirt virtualization API (already included)
    qemu-kvm \            # Install QEMU virtualization software (already included)
    libvirt-daemon-system \ # Enable libvirt daemon at boot (already included)
    libvirt-clients \     # Install libvirt clients (already included)
    virt-manager \        # Install Virtual Machine Manager GUI (already included)
    virt-viewer \         # Install Virtual Machine Viewer GUI (already included)
    qemu-system \         # Install QEMU virtualization system (already included)
    libosinfo-bin \       # Install libosinfo library (already included)
    libvirt-dev \         # Install libvirt development files (already included)
    libvirt-doc \         # Install libvirt documentation (already included)
    libvirt-utils \       # Install libvirt utilities (already included)
    libvirt-python \      # Install libvirt Python bindings (already included)
    libvirt-python3 \     # Install libvirt Python 3 bindings (already included)
    python-libvirt \      # Install Python libvirt library (already included)
    python3-libvirt \     # Install Python 3 libvirt library (already included)
    nano \               # Keep nano text editor (already included)
    htop \                # Keep htop process viewer (already included)
    curl \                # Keep curl (already included)
    socat \               # Keep socat (already included)
    net-tools \           # Keep net-tools (already included)
    nload \               # Keep nload network traffic monitor
    iftop \               # Keep iftop network traffic analyzer
    iotop \               # Keep iotop I/O traffic monitor
    glances \             # Keep glances system monitoring tool
    nethogs \             # Keep nethogs network traffic monitor
    vnstat \              # Keep vnstat network traffic monitor
    mtr \                 # Keep mtr network diagnostic tool
    nmap \                # Keep nmap network scanner
    tcpdump \             # Keep tcpdump network packet analyzer
    wireshark \           # Keep wireshark network protocol analyzer
    netcat \              # Keep netcat network debugging tool
    iperf3 \              # Keep iperf3 network performance measurement tool
    fping \               # Keep fping network latency measurement tool
    traceroute \          # Keep traceroute network tracing tool
    dnsutils \            # Keep dnsutils DNS utilities

# Confirm or cancel the reboot
while true; do
    read -p "Do you want to reboot the server? (y/n): " yn
    case $yn in
        [Yy]* ) sudo reboot; break;;
        [Nn]* ) echo "Reboot aborted."; exit;;
        * ) echo "Please answer yes or no.";;
    esac
done
