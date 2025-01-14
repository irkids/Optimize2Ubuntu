#!/usr/bin/env python3

import os
import subprocess
import sys
import shutil
from pathlib import Path
from logging import getLogger, basicConfig, INFO

basicConfig(level=INFO)
logger = getLogger("IKEv2 Installer")

def check_root():
   if os.geteuid() != 0:
       logger.error("This script must be run as root!")
       sys.exit(1)

check_root()

def run_command(cmd):
   try:
       result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
       return result.stdout.strip()
   except subprocess.CalledProcessError as e:
       logger.error(f"Command failed: {e.cmd}\nError: {e.stderr}")
       return None

class IPSecManager:
   def __init__(self):
       self.ipsec_dir = "/etc/ipsec.d"
       self.secrets_file = "/etc/ipsec.secrets"
       
   def add_user(self, username, password):
       try:
           run_command(f"ipsec pki --gen --outform pem > {self.ipsec_dir}/{username}_key.pem")
           run_command(f"ipsec pki --pub --in {self.ipsec_dir}/{username}_key.pem | "
                      f"ipsec pki --issue --cacert {self.ipsec_dir}/cacerts/ca.crt "
                      f"--cakey {self.ipsec_dir}/private/ca.key "
                      f'--dn "C=IR, O=IRSSH, CN={username}" '
                      f"--outform pem > {self.ipsec_dir}/certs/{username}.crt")
           
           with open(self.secrets_file, 'a') as f:
               f.write(f'{username} : EAP "{password}"\n')
           
           run_command("ipsec reload")
           return True
           
       except Exception as e:
           logger.error(f"Failed to add user {username}: {str(e)}")
           return False
   
   def remove_user(self, username):
       try:
           key_file = f"{self.ipsec_dir}/{username}_key.pem"
           cert_file = f"{self.ipsec_dir}/certs/{username}.crt"
           if os.path.exists(key_file):
               os.remove(key_file)
           if os.path.exists(cert_file):
               os.remove(cert_file)
           
           with open(self.secrets_file, 'r') as f:
               lines = f.readlines()
           with open(self.secrets_file, 'w') as f:
               for line in lines:
                   if not line.startswith(f'{username} : EAP'):
                       f.write(line)
           
           run_command("ipsec reload")
           return True
           
       except Exception as e:
           logger.error(f"Failed to remove user {username}: {str(e)}")
           return False
   
   def list_users(self):
       try:
           users = []
           with open(self.secrets_file, 'r') as f:
               for line in f:
                   if ' : EAP "' in line:
                       username = line.split(':')[0].strip()
                       users.append(username)
           return users
       except Exception as e:
           logger.error(f"Failed to list users: {str(e)}")
           return []

def main():
   manager = IPSecManager()
   
   if len(sys.argv) < 2:
       print("Usage:")
       print("  Add user:    ikev2.py add-user <username> <password>")
       print("  Remove user: ikev2.py remove-user <username>")
       print("  List users:  ikev2.py list-users")
       sys.exit(1)
   
   command = sys.argv[1]
   
   if command == "add-user" and len(sys.argv) == 4:
       username, password = sys.argv[2], sys.argv[3]
       if manager.add_user(username, password):
           print(f"User {username} added successfully")
       else:
           sys.exit(1)
           
   elif command == "remove-user" and len(sys.argv) == 3:
       username = sys.argv[2]
       if manager.remove_user(username):
           print(f"User {username} removed successfully")
       else:
           sys.exit(1)
           
   elif command == "list-users":
       users = manager.list_users()
       print("\n".join(users))
       
   else:
       print("Invalid command")
       sys.exit(1)

if __name__ == "__main__":
   main()
