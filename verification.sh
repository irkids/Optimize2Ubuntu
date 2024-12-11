#!/usr/bin/env bash

# Robust Bash Verification Script
# Enhanced error handling and syntax compatibility

# Strict mode with additional error handling
set -euo pipefail

# Comprehensive error handling function
handle_error() {
    local lineno="$1"
    local command="$2"
    local exitcode="$3"
    
    echo "Error: Command '$command' failed at line $lineno with exit code $exitcode" >&2
}

# Trap errors with detailed logging
trap 'handle_error ${LINENO} "$BASH_COMMAND" $?' ERR

# Color Definitions
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m' # No Color

# Logging Function
log() {
    local level="$1"
    local message="$2"
    local timestamp
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    
    case "$level" in
        "INFO")
            echo -e "${GREEN}[INFO] ${timestamp}: ${message}${NC}"
            ;;
        "WARNING")
            echo -e "${YELLOW}[WARNING] ${timestamp}: ${message}${NC}" >&2
            ;;
        "ERROR")
            echo -e "${RED}[ERROR] ${timestamp}: ${message}${NC}" >&2
            ;;
        *)
            echo -e "[${level}] ${timestamp}: ${message}"
            ;;
    esac
}

# Dependency Checker
check_dependencies() {
    local dependencies=(
        "curl"
        "wget"
        "git"
        "python3"
        "node"
        "npm"
        "php"
        "perl"
    )
    
    local missing_deps=()
    
    for dep in "${dependencies[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        log "ERROR" "Missing dependencies: ${missing_deps[*]}"
        return 1
    fi
    
    log "INFO" "All dependencies are available"
    return 0
}

# Repository Validation Function
validate_github_repo() {
    local repo_url="$1"
    
    # Enhanced GitHub URL validation
    if [[ ! "$repo_url" =~ ^https://github\.com/[a-zA-Z0-9-]+/[a-zA-Z0-9_-]+(/.*)?$ ]]; then
        log "ERROR" "Invalid GitHub repository URL format"
        return 1
    fi
    
    # Validate repository accessibility
    if ! git ls-remote "$repo_url" &> /dev/null; then
        log "ERROR" "Cannot access repository. Check URL or network connectivity."
        return 1
    fi
    
    log "INFO" "Repository URL validated successfully"
    return 0
}

# Function Extraction
extract_functions() {
    local script_path="$1"
    local language="$2"
    
    case "$language" in
        "bash")
            grep -E '^[[:space:]]*[a-zA-Z_][a-zA-Z0-9_]*\(\)' "$script_path" | \
                sed -E 's/[[:space:]]*([a-zA-Z_][a-zA-Z0-9_]*)\(\).*/\1/'
            ;;
        "python")
            grep -E '^def [a-zA-Z_][a-zA-Z0-9_]*\(' "$script_path" | \
                sed -E 's/def ([a-zA-Z_][a-zA-Z0-9_]*)\(.*/\1/'
            ;;
        "javascript")
            grep -E '(function[[:space:]]+[a-zA-Z_][a-zA-Z0-9_]*\(|[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*function\()' "$script_path" | \
                sed -E 's/.*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(.*/\1/'
            ;;
        "php")
            grep -E '^(public |private |protected )?function [a-zA-Z_][a-zA-Z0-9_]*\(' "$script_path" | \
                sed -E 's/.*function ([a-zA-Z_][a-zA-Z0-9_]*)\(.*/\1/'
            ;;
        *)
            log "ERROR" "Unsupported language: $language"
            return 1
            ;;
    esac
}

# Main Verification Function
verify_script() {
    local repo_url="$1"
    local language="${2:-bash}"
    
    # Validate repository
    if ! validate_github_repo "$repo_url"; then
        return 1
    fi
    
    # Create temporary directory
    local temp_dir
    temp_dir=$(mktemp -d)
    
    # Clone repository
    if ! git clone --depth 1 "$repo_url" "$temp_dir"; then
        log "ERROR" "Failed to clone repository"
        return 1
    fi
    
    # Find script files
    local script_files
    mapfile -t script_files < <(find "$temp_dir" -type f -name "*.$language")
    
    if [ ${#script_files[@]} -eq 0 ]; then
        log "ERROR" "No $language scripts found in repository"
        return 1
    fi
    
    # Verify functions
    for script in "${script_files[@]}"; do
        log "INFO" "Analyzing script: $script"
        
        # Extract functions
        local functions
        mapfile -t functions < <(extract_functions "$script" "$language")
        
        if [ ${#functions[@]} -eq 0 ]; then
            log "WARNING" "No functions found in $script"
            continue
        fi
        
        log "INFO" "Found ${#functions[@]} functions"
        
        # Optional: Add function verification logic here
    done
    
    # Cleanup
    rm -rf "$temp_dir"
    
    log "INFO" "Script verification completed"
}

# Main Execution
main() {
    # Check dependencies
    if ! check_dependencies; then
        log "ERROR" "Dependency check failed"
        exit 1
    fi
    
    # Ensure at least repository URL is provided
    if [ $# -lt 1 ]; then
        log "ERROR" "Usage: $0 <github_repo_url> [language]"
        exit 1
    fi
    
    local repo_url="$1"
    local language="${2:-bash}"
    
    # Run verification
    verify_script "$repo_url" "$language"
}

# Only run main if script is being executed, not sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
