#!/bin/bash

apt update && apt upgrade -y
echo "nameserver 8.8.8.8" |
 sudo tee -a /etc/resolv.conf
echo "nameserver 8.8.4.4" |
 sudo tee -a /etc/resolv.conf
sudo ifconfig eth0 mtu 1400
sudo apt-get install pkg-config libssl-dev -y
apt install curl socat -y
apt install -y htop iotop net-tools nmap
sudo NEETRESTART_MODE=a apt-get update --yes

if [[ $(uname -r | cut -d. -f1) -lt 4 ]]; then
  echo "Your kernel does not support TCP Fast Open!"
  exit 1
fi

echo "net.ipv4.tcp_fastopen=3" |
 sudo tee -a /etc/sysctl.conf
sudo sysctl -p
sysctl net.ipv4.tcp_fastopen
sudo apt install fail2ban -y
sudo systemctl enable fail2ban
sudo service ssh restart
sudo apt install rkhunter -y
sudo rkhunter --check
sudo apt-get -y install software-properties-common apt-get install -y stunnel4 cmake screenfetch openssl
sudo add-apt-repository ppa:ondrej/php -y
apt-get install apache2 zip unzip net-tools curl mariadb-server -y
apt-get install php php-cli php-mbstring php-dom php-pdo php-mysql -y
apt-get install npm -y
sudo apt-get install coreutils
apt install php8.1 php8.1-mysql php8.1-xml php8.1-curl cron -y
apt install php8.2 php8.2-mysql php8.2-xml php8.2-curl cron -y
apt install php8.3 php8.3-mysql php8.3-xml php8.3-curl cron -y
sudo apt-get install git -y
sudo apt install ufw nginx curl jq python3-pip python3-setuptools
sudo apt install perl -y
sudo apt install libdbd-mysql-perl -y
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install Python3.12 -y
sudo apt install python -y
sudo apt-get install nano
sudo apt-get install ufw
sudo apt-get install nginx -y
sudo systemctl enable nginx
apt install tasksel -y
sudo tasksel install lamp-server
systemctl reload sshd
systemctl restart sshd
service ssh reload
service ssh restart


echo "Optimize2Ubuntu Installed!"


echo " Restart The Server To Optimize Ubuntu Server "

echo -e "Warning: This command will reboot your server."

echo -e "Do you want to reboot your server? (Y/N)"
read answer

if [ "$answer" = "y" ]; then
  echo -e "Server rebooting ..."
  sudo reboot
else
  echo -e "Not reboot."
fi
