#!/usr/bin/env bash

# Ultra-Advanced Script Function Verifier and Installer
# Version 2.2 - Comprehensive Multi-Language Script Verification Utility

# Strict error handling and debugging
set -euo pipefail
IFS=$'\n\t'

# Color Codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Global Configuration
SCRIPT_NAME="$(basename "$0")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_DIR="/tmp/script_verifier_$(date +%Y%m%d_%H%M%S)"
CONFIG_FILE="${SCRIPT_DIR}/script_verifier.conf"

# Ensure temporary directory exists
mkdir -p "$TEMP_DIR"

# Log File Paths
LOG_FILE="${TEMP_DIR}/verification.log"
DEPENDENCY_LOG="${TEMP_DIR}/dependency_check.log"

# Logging and Error Handling
log_message() {
    local log_level="$1"
    local message="$2"
    
    # Ensure log directory exists
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Log to file and stdout
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [$log_level] $message" | tee -a "$LOG_FILE"
}

error_exit() {
    local error_message="$1"
    log_message "ERROR" "$error_message"
    echo -e "${RED}FATAL ERROR: $error_message${NC}" >&2
    cleanup
    exit 1
}

# Prerequisite Checking and Installation
check_bash_version() {
    local required_major=4
    local current_version="${BASH_VERSION%.*}"
    
    if (( $(echo "$current_version >= $required_major" | bc -l) )); then
        log_message "INFO" "Bash version $current_version meets requirements"
    else
        error_exit "Bash version must be 4.x or higher. Current version: $current_version"
    fi
}

install_language_prerequisites() {
    log_message "INFO" "Checking and installing language prerequisites..."

    # Ensure package lists are updated
    apt-get update -qq

    # Python check and installation
    if ! command -v python3 &> /dev/null; then
        log_message "WARN" "Python 3 not found. Attempting to install..."
        apt-get install -y python3 python3-pip || error_exit "Failed to install Python"
    fi

    # Node.js and npm check and installation
    if ! command -v node &> /dev/null || ! command -v npm &> /dev/null; then
        log_message "WARN" "Node.js or npm not found. Attempting to install..."
        curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -
        apt-get install -y nodejs || error_exit "Failed to install Node.js"
    fi

    # Install required Node.js libraries
    npm install -g acorn acorn-walk || error_exit "Failed to install Node.js libraries"

    # PHP check and installation
    if ! command -v php &> /dev/null; then
        log_message "WARN" "PHP not found. Attempting to install..."
        apt-get install -y php || error_exit "Failed to install PHP"
    fi

    # Perl check and installation
    if ! command -v perl &> /dev/null; then
        log_message "WARN" "Perl not found. Attempting to install..."
        apt-get install -y perl || error_exit "Failed to install Perl"
    fi

    # Install curl if not present
    if ! command -v curl &> /dev/null; then
        log_message "WARN" "curl not found. Attempting to install..."
        apt-get install -y curl || error_exit "Failed to install curl"
    fi

    log_message "INFO" "All language prerequisites installed successfully"
}

# Script Verification Function
verify_script() {
    local script_url="$1"
    local script_language="$2"
    local download_path="${TEMP_DIR}/downloaded_script.${script_language}"

    log_message "INFO" "Downloading script from $script_url"
    
    # Download the script
    if ! curl -fsSL -o "$download_path" "$script_url"; then
        error_exit "Failed to download script from $script_url"
    fi

    log_message "INFO" "Verifying $script_language script syntax"

    # Language-specific verification
    case "$script_language" in
        "bash")
            if ! bash -n "$download_path"; then
                error_exit "Bash script syntax check failed"
            fi
            ;;
        "python")
            if ! python3 -m py_compile "$download_path"; then
                error_exit "Python script syntax check failed"
            fi
            ;;
        "javascript")
            if ! node -c "$download_path"; then
                error_exit "JavaScript script syntax check failed"
            fi
            ;;
        "php")
            if ! php -l "$download_path"; then
                error_exit "PHP script syntax check failed"
            fi
            ;;
        "perl")
            if ! perl -c "$download_path"; then
                error_exit "Perl script syntax check failed"
            fi
            ;;
        *)
            error_exit "Unsupported script language: $script_language"
            ;;
    esac

    log_message "INFO" "Script syntax verification successful"
    
    # Optional: Additional security checks can be added here
    echo -e "${GREEN}Script verification completed successfully!${NC}"
}

# Main Execution
main() {
    # Ensure root/sudo capabilities for system-wide installations
    if [[ $EUID -ne 0 ]]; then
        error_exit "This script must be run with sudo/root privileges"
    fi

    # Check bash version
    check_bash_version

    # Install prerequisites
    install_language_prerequisites

    # Interactive script verification
    echo -e "${YELLOW}Ultra-Advanced Script Function Verifier${NC}"
    read -p "Enter GitHub RAW script URL: " script_url
    
    PS3="Select script language: "
    options=("Bash" "Python" "JavaScript" "PHP" "Perl" "Quit")
    select opt in "${options[@]}"
    do
        case $opt in
            "Bash") verify_script "$script_url" "bash"; break ;;
            "Python") verify_script "$script_url" "python"; break ;;
            "JavaScript") verify_script "$script_url" "javascript"; break ;;
            "PHP") verify_script "$script_url" "php"; break ;;
            "Perl") verify_script "$script_url" "perl"; break ;;
            "Quit") exit 0 ;;
            *) echo "Invalid option $REPLY" ;;
        esac
    done
}

# Cleanup Function
cleanup() {
    [[ -d "$TEMP_DIR" ]] && rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# Script Entry Point
main "$@"
