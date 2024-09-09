#!/bin/bash

sudo apt-get update -y && sudo apt-get dist-upgrade -y
sudo apt-get install pkg-config libssl-dev -y
apt install curl socat -y
sudo apt-get -y install software-properties-common apt-get install -y stunnel4 cmake screenfetch openssl
sudo add-apt-repository ppa:ondrej/php -y
sudo apt-get -y install software-properties-common
apt-get install apache2 zip unzip net-tools curl mariadb-server -y
apt-get install php php-cli php-mbstring php-dom php-pdo php-mysql -y
apt-get install npm -y
sudo apt-get install coreutils
apt install php8.1 php8.1-mysql php8.1-xml php8.1-curl cron -y
apt install php8.2 php8.2-mysql php8.2-xml php8.2-curl cron -y
apt install php8.3 php8.3-mysql php8.3-xml php8.3-curl cron -y
sudo apt-get install php8. php-pear -y
sudo apt install php8.1 php8.1-fpm php8.1-mysql php8.1-curl php8.1-gd php8.1-mbstring php8.1-xml php8.1-xmlrpc php8.1-zip -y
sudo apt install iftop -y
sudo apt install perl -y
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get install nano
sudo apt-get install ufw
sudo apt-get install nginx -y
sudo systemctl enable nginx
bash <(curl -Ls https://raw.githubusercontent.com/irkids/Optimize2Ubuntu.sh/main/TCP2OpT --ipv4)

systemctl reload sshd
systemctl restart sshd
service ssh reload
service ssh restart

# Call the reboot script
if source /path/to/reboot_script.sh; then
  read -p "Do you want to reboot the server? (y/n): " choice

  case "$choice" in 
    y|Y ) sudo reboot;;
    n|N ) echo "Reboot cancelled.";;
    * ) echo "Invalid option. Please enter y or n.";;
  esac
else
  echo "Failed to source the reboot script."
fi
