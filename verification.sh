#!/usr/bin/env bash

# Ultra Advanced Script Verification and Installation Tool
# Supports comprehensive multi-language script analysis and verification

# Color Codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration and Paths
VERSION="1.3.0"
CONFIG_DIR="${HOME}/.script-verifier"
TEMP_DIR="/tmp/script-verifier-$(date +%s)"
LOG_FILE="${CONFIG_DIR}/verification.log"
PREREQUISITES_FILE="${CONFIG_DIR}/prerequisites.json"

# Ensure minimum BASH version
if [[ ${BASH_VERSINFO[0]} -lt 4 ]]; then
    echo -e "${RED}Error: Requires BASH 4.x or higher. Current version: ${BASH_VERSION}${NC}"
    exit 1
}

# Logging Function
log_message() {
    local log_level="${1:-INFO}"
    local message="${2}"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [${log_level}] ${message}" | tee -a "${LOG_FILE}"
}

# Prerequisite Checking and Installation
check_and_install_prerequisites() {
    # Install jq if not present
    if ! command -v jq &> /dev/null; then
        echo -e "${YELLOW}Installing jq...${NC}"
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y jq
        elif command -v yum &> /dev/null; then
            sudo yum install -y jq
        elif command -v brew &> /dev/null; then
            brew install jq
        else
            echo -e "${RED}Cannot install jq. Please install it manually.${NC}"
            return 1
        fi
    fi

    # Create prerequisites configuration if not exists
    mkdir -p "$(dirname "${PREREQUISITES_FILE}")"
    if [[ ! -f "${PREREQUISITES_FILE}" ]]; then
        cat > "${PREREQUISITES_FILE}" << EOL
{
    "languages": {
        "python": {
            "min_version": "3.7",
            "install_command": "sudo apt-get install -y python3 python3-pip"
        },
        "nodejs": {
            "min_version": "14.0.0",
            "install_command": "curl -fsSL https://deb.nodesource.com/setup_14.x | sudo -E bash - && sudo apt-get install -y nodejs",
            "npm_libraries": ["acorn", "acorn-walk"]
        },
        "php": {
            "min_version": "7.4",
            "install_command": "sudo apt-get install -y php php-cli"
        },
        "perl": {
            "min_version": "5.26",
            "install_command": "sudo apt-get install -y perl"
        }
    },
    "tools": {
        "curl": {
            "install_command": "sudo apt-get install -y curl"
        },
        "wget": {
            "install_command": "sudo apt-get install -y wget"
        }
    }
}
EOL
    fi

    # Prerequisite Checking Function
    check_prerequisite() {
        local tool_or_language="$1"
        local min_version="${2:-}"
        
        case "$tool_or_language" in
            "python")
                if command -v python3 &> /dev/null; then
                    local py_version=$(python3 --version | cut -d' ' -f2)
                    if [[ "$(printf '%s\n' "$min_version" "$py_version" | sort -V | head -n1)" == "$min_version" ]]; then
                        echo -e "${GREEN}✓ Python ${py_version} installed${NC}"
                        return 0
                    fi
                fi
                ;;
            "nodejs")
                if command -v node &> /dev/null; then
                    local node_version=$(node --version | sed 's/v//')
                    if [[ "$(printf '%s\n' "$min_version" "$node_version" | sort -V | head -n1)" == "$min_version" ]]; then
                        echo -e "${GREEN}✓ Node.js ${node_version} installed${NC}"
                        # Check npm libraries
                        npm list acorn acorn-walk &> /dev/null || {
                            npm install acorn acorn-walk
                        }
                        return 0
                    fi
                fi
                ;;
            "php")
                if command -v php &> /dev/null; then
                    local php_version=$(php --version | head -n 1 | cut -d' ' -f2)
                    if [[ "$(printf '%s\n' "$min_version" "$php_version" | sort -V | head -n1)" == "$min_version" ]]; then
                        echo -e "${GREEN}✓ PHP ${php_version} installed${NC}"
                        return 0
                    fi
                fi
                ;;
            "perl")
                if command -v perl &> /dev/null; then
                    local perl_version=$(perl -V | grep 'This is perl' | cut -d' ' -f4)
                    if [[ "$(printf '%s\n' "$min_version" "$perl_version" | sort -V | head -n1)" == "$min_version" ]]; then
                        echo -e "${GREEN}✓ Perl ${perl_version} installed${NC}"
                        return 0
                    fi
                fi
                ;;
            "curl"|"wget")
                command -v "$tool_or_language" &> /dev/null
                return $?
                ;;
        esac
        
        return 1
    }

    # Read Prerequisites JSON Safely
    read_json_value() {
        local json_file="$1"
        local query="$2"
        jq -r "$query" "$json_file" 2>/dev/null
    }

    # Main Prerequisite Checking and Installation
    # Languages
    local language_keys=$(jq -r '.languages | keys[]' "${PREREQUISITES_FILE}" 2>/dev/null)
    for lang in $language_keys; do
        min_version=$(jq -r ".languages.${lang}.min_version" "${PREREQUISITES_FILE}" 2>/dev/null)
        install_command=$(jq -r ".languages.${lang}.install_command" "${PREREQUISITES_FILE}" 2>/dev/null)
        
        check_prerequisite "$lang" "$min_version" || {
            echo -e "${YELLOW}Installing ${lang}...${NC}"
            eval "$install_command"
        }
    done

    # Tools
    local tool_keys=$(jq -r '.tools | keys[]' "${PREREQUISITES_FILE}" 2>/dev/null)
    for tool in $tool_keys; do
        install_command=$(jq -r ".tools.${tool}.install_command" "${PREREQUISITES_FILE}" 2>/dev/null)
        
        check_prerequisite "$tool" || {
            echo -e "${YELLOW}Installing ${tool}...${NC}"
            eval "$install_command"
        }
    done
}

# Validate GitHub or Raw GitHub URL
validate_script_url() {
    local url="$1"
    
    # Regex for GitHub repository or raw content URLs
    local github_repo_regex='^https?://(www\.)?github\.com/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+(\.git)?$'
    local raw_github_regex='^https?://raw\.githubusercontent\.com/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+/.+\.(py|js|php|pl)$'
    
    if [[ "$url" =~ $github_repo_regex ]] || [[ "$url" =~ $raw_github_regex ]]; then
        echo "$url"
        return 0
    else
        echo -e "${RED}Invalid GitHub repository or raw script URL${NC}"
        return 1
    fi
}

# Download Script
download_script() {
    local url="$1"
    local output_dir="${TEMP_DIR}/script"
    
    mkdir -p "$output_dir"
    
    # Detect download method
    if command -v curl &> /dev/null; then
        curl -L "$url" -o "${output_dir}/script" 2>/dev/null
    elif command -v wget &> /dev/null; then
        wget -O "${output_dir}/script" "$url" 2>/dev/null
    else
        echo -e "${RED}Neither curl nor wget available for download${NC}"
        return 1
    fi
    
    # Check download success
    if [[ -s "${output_dir}/script" ]]; then
        echo "${output_dir}/script"
        return 0
    else
        echo -e "${RED}Failed to download script${NC}"
        return 1
    fi
}

# Function Verification (Simplified for this example)
verify_script_functions() {
    local script_path="$1"
    local language=$(file -b --mime-type "$script_path" | cut -d'/' -f2)
    
    echo -e "${YELLOW}Analyzing script: $script_path${NC}"
    
    case "$language" in
        "x-python")
            functions=$(grep -E '^def ' "$script_path" | awk '{print $2}' | cut -d'(' -f1)
            ;;
        "x-javascript")
            functions=$(grep -E '(function |=>)' "$script_path" | awk '{print $2}' | cut -d'(' -f1)
            ;;
        "x-php")
            functions=$(grep -E '^(public|private|protected)? function ' "$script_path" | awk '{print $3}' | cut -d'(' -f1)
            ;;
        "x-perl")
            functions=$(grep -E '^sub ' "$script_path" | awk '{print $2}')
            ;;
        *)
            echo -e "${RED}Unsupported language: $language${NC}"
            return 1
            ;;
    esac
    
    echo -e "${YELLOW}Detected Functions:${NC}"
    for func in $functions; do
        echo -e "${GREEN}✓ ${func}${NC}"
    done
}

# Main Execution
main() {
    # Initialize directories
    mkdir -p "${CONFIG_DIR}"
    mkdir -p "${TEMP_DIR}"

    # Check and install prerequisites
    check_and_install_prerequisites

    # Interactive URL input
    read -p "Enter GitHub repository or raw script URL: " script_url

    # Validate URL
    validated_url=$(validate_script_url "$script_url")
    if [[ $? -ne 0 ]]; then
        echo -e "${RED}Invalid URL. Exiting.${NC}"
        exit 1
    fi

    # Download script
    downloaded_script=$(download_script "$validated_url")
    if [[ $? -ne 0 ]]; then
        echo -e "${RED}Script download failed. Exiting.${NC}"
        exit 1
    fi

    # Verify script functions
    verify_script_functions "$downloaded_script"
}

# Run main
main "$@"

# Cleanup
cleanup() {
    rm -rf "${TEMP_DIR}"
}
trap cleanup EXIT

exit 0
