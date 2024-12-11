#!/usr/bin/env bash

# Ultra-Advanced Script Function Verifier and Installer
# Version 2.0 - Comprehensive Multi-Language Script Verification Utility

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
LOG_FILE="${TEMP_DIR}/verification.log"
DEPENDENCY_LOG="${TEMP_DIR}/dependency_check.log"

# Logging and Error Handling
log_message() {
    local log_level="$1"
    local message="$2"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [$log_level] $message" | tee -a "$LOG_FILE"
}

error_exit() {
    log_message "ERROR" "$1"
    echo -e "${RED}FATAL ERROR: $1${NC}" >&2
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

    # Python check and installation
    if ! command -v python3 &> /dev/null; then
        log_message "WARN" "Python 3 not found. Attempting to install..."
        sudo apt-get update
        sudo apt-get install -y python3 python3-pip || error_exit "Failed to install Python"
    fi

    # Node.js and npm check and installation
    if ! command -v node &> /dev/null || ! command -v npm &> /dev/null; then
        log_message "WARN" "Node.js or npm not found. Attempting to install..."
        curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
        sudo apt-get install -y nodejs || error_exit "Failed to install Node.js"
    fi

    # Install required Node.js libraries
    npm install acorn acorn-walk || error_exit "Failed to install Node.js libraries"

    # PHP check and installation
    if ! command -v php &> /dev/null; then
        log_message "WARN" "PHP not found. Attempting to install..."
        sudo apt-get install -y php || error_exit "Failed to install PHP"
    fi

    # Perl check and installation
    if ! command -v perl &> /dev/null; then
        log_message "WARN" "Perl not found. Attempting to install..."
        sudo apt-get install -y perl || error_exit "Failed to install Perl"
    fi

    # Install curl or wget if not present
    if ! command -v curl &> /dev/null && ! command -v wget &> /dev/null; then
        log_message "WARN" "curl and wget not found. Attempting to install curl..."
        sudo apt-get install -y curl || error_exit "Failed to install curl"
    fi

    log_message "INFO" "All language prerequisites installed successfully"
}

# Advanced Function Extraction and Verification
extract_functions_advanced() {
    local script_path="$1"
    local language="$2"

    case "$language" in
        "bash")
            bash_function_extractor "$script_path"
            ;;
        "python")
            python_function_extractor "$script_path"
            ;;
        "javascript")
            javascript_function_extractor "$script_path"
            ;;
        "php")
            php_function_extractor "$script_path"
            ;;
        "perl")
            perl_function_extractor "$script_path"
            ;;
        *)
            error_exit "Unsupported language: $language"
            ;;
    esac
}

bash_function_extractor() {
    local script_path="$1"
    grep -E '^[[:space:]]*[a-zA-Z_][a-zA-Z0-9_]*\(\)' "$script_path" | \
        sed -E 's/[[:space:]]*([a-zA-Z_][a-zA-Z0-9_]*)\(\).*/\1/'
}

python_function_extractor() {
    local script_path="$1"
    python3 -c "
import ast
import sys

def extract_functions(file_path):
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read())
    
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    print('\n'.join(functions))

extract_functions('$script_path')
"
}

javascript_function_extractor() {
    local script_path="$1"
    node -e "
const acorn = require('acorn');
const walk = require('acorn-walk');
const fs = require('fs');

const code = fs.readFileSync('$script_path', 'utf8');
const ast = acorn.parse(code, { ecmaVersion: 2020 });

const functions = new Set();
walk.simple(ast, {
    FunctionDeclaration(node) {
        if (node.id && node.id.name) functions.add(node.id.name);
    },
    VariableDeclarator(node) {
        if (node.id && node.id.type === 'Identifier' && 
            node.init && node.init.type === 'FunctionExpression') {
            functions.add(node.id.name);
        }
    }
});

console.log(Array.from(functions).join('\n'));
"
}

php_function_extractor() {
    local script_path="$1"
    php -r "
    \$code = file_get_contents('$script_path');
    preg_match_all('/function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(/', \$code, \$matches);
    echo implode('\n', \$matches[1]);
"
}

perl_function_extractor() {
    local script_path="$1"
    perl -ne 'print $1 if /^sub\s+([a-zA-Z_][a-zA-Z0-9_]*)/' "$script_path"
}

# Advanced Verification with Detailed Error Analysis
verify_function_advanced() {
    local function_name="$1"
    local script_path="$2"
    local language="$3"

    case "$language" in
        "bash")
            bash_function_verification "$function_name"
            ;;
        "python")
            python_function_verification "$script_path" "$function_name"
            ;;
        "javascript")
            javascript_function_verification "$script_path" "$function_name"
            ;;
        "php")
            php_function_verification "$script_path" "$function_name"
            ;;
        "perl")
            perl_function_verification "$script_path" "$function_name"
            ;;
    esac
}

bash_function_verification() {
    local function_name="$1"
    if declare -f "$function_name" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

python_function_verification() {
    local script_path="$1"
    local function_name="$2"
    python3 -c "
import importlib.util
import sys

spec = importlib.util.spec_from_file_location('script_module', '$script_path')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

if hasattr(module, '$function_name'):
    sys.exit(0)
else:
    sys.exit(1)
"
}

javascript_function_verification() {
    local script_path="$1"
    local function_name="$2"
    node -e "
const fs = require('fs');
const script = require('$script_path');

if (typeof script['$function_name'] === 'function') {
    process.exit(0);
} else {
    process.exit(1);
}
"
}

php_function_verification() {
    local script_path="$1"
    local function_name="$2"
    php -r "
include '$script_path';
if (function_exists('$function_name')) {
    exit(0);
} else {
    exit(1);
}
"
}

perl_function_verification() {
    local script_path="$1"
    local function_name="$2"
    perl -I. -M"$script_path" -e "
if (defined(&$function_name)) {
    exit(0);
} else {
    exit(1);
}
"
}

# Script Verification Workflow
verify_script() {
    local script_url="$1"
    local language="$2"

    # Download script
    local script_path="${TEMP_DIR}/source_script.${language}"
    download_script "$script_url" "$script_path"

    # Verify file permissions
    verify_file_permissions "$script_path"

    # Extract functions
    local functions=()
    mapfile -t functions < <(extract_functions_advanced "$script_path" "$language")

    # Results tracking
    local total_functions=${#functions[@]}
    local verified_functions=0
    local failed_functions=0

    # Function verification
    for func in "${functions[@]}"; do
        if verify_function_advanced "$func" "$script_path" "$language"; then
            echo -e "${GREEN}✓ Function '$func' verified successfully${NC}"
            ((verified_functions++))
        else
            echo -e "${RED}✗ Function '$func' verification failed${NC}"
            ((failed_functions++))
            
            # Detailed failure analysis
            analyze_function_failure "$func" "$script_path" "$language"
        fi
    done

    # Final summary
    echo -e "\n${BLUE}Verification Summary${NC}"
    echo -e "Total Functions: $total_functions"
    echo -e "Verified Functions: $verified_functions"
    echo -e "Failed Functions: $failed_functions"
}

download_script() {
    local script_url="$1"
    local script_path="$2"

    if command -v curl &> /dev/null; then
        curl -fsSL "$script_url" -o "$script_path"
    elif command -v wget &> /dev/null; then
        wget -q "$script_url" -O "$script_path"
    else
        error_exit "Neither curl nor wget available for script download"
    fi

    if [[ ! -s "$script_path" ]]; then
        error_exit "Failed to download script from $script_url"
    fi
}

verify_file_permissions() {
    local script_path="$1"
    local permissions
    permissions=$(stat -c "%a" "$script_path")

    if [[ "$permissions" =~ ^[0-7]{3}$ ]]; then
        log_message "INFO" "File permissions: $permissions"
    else
        error_exit "Invalid file permissions detected"
    fi
}

analyze_function_failure() {
    local function_name="$1"
    local script_path="$2"
    local language="$3"

    log_message "WARN" "Analyzing failure for function: $function_name"
    
    # Advanced failure detection strategies
    case "$language" in
        "python")
            python3 -m py_compile "$script_path" || log_message "ERROR" "Python syntax error in script"
            ;;
        "javascript")
            node --check "$script_path" || log_message "ERROR" "JavaScript syntax error in script"
            ;;
        "php")
            php -l "$script_path" || log_message "ERROR" "PHP syntax error in script"
            ;;
        "perl")
            perl -c "$script_path" || log_message "ERROR" "Perl syntax error in script"
            ;;
    esac
}

# Cleanup Function
cleanup() {
    [[ -d "$TEMP_DIR" ]] && rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# Main Execution
main() {
    # Ensure root/sudo capabilities for system-wide installations
    [[ $EUID -ne 0 ]] && error_exit "This script must be run with sudo/root privileges"

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

# Script Entry Point
main "$@"
