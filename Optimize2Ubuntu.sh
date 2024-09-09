#!/bin/bash

sudo apt-get update -y && sudo apt-get dist-upgrade -y
sudo apt-get install pkg-config libssl-dev -y
apt install curl socat -y
sudo apt-get install jq -y
sudo apt install openssh-server -y
sudo service ssh restart
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
sudo apt-get install git -y
sudo apt install iftop -y
sudo apt install perl -y
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python -y
sudo apt-get install nano
sudo apt-get install ufw
sudo apt-get install nginx -y
sudo systemctl enable nginx
apt install tasksel -y
sudo tasksel install lamp-server
bash <(curl -Ls https://raw.githubusercontent.com/irkids/Optimize2Ubuntu.sh/main/TCP2OpT --ipv4)
sudo apt update && sudo apt full-upgrade -y
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
