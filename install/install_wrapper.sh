#!/bin/bash
#
# Linucast Download and Run Script
# This script downloads and runs the installer

set -e

# Detect if curl or wget is available
if command -v curl &> /dev/null; then
    DOWNLOADER="curl -fsSL"
    DOWNLOADER_OUTPUT="-o"
elif command -v wget &> /dev/null; then
    DOWNLOADER="wget -q"
    DOWNLOADER_OUTPUT="-O"
else
    echo "Error: Neither curl nor wget is installed. Please install one of them and try again."
    exit 1
fi

echo "Downloading Linucast installer..."
$DOWNLOADER https://raw.githubusercontent.com/yourusername/linucast/main/install/one_line_install.sh $DOWNLOADER_OUTPUT /tmp/linucast_installer.sh

chmod +x /tmp/linucast_installer.sh
echo "Running installer..."
/tmp/linucast_installer.sh "$@" # Pass any arguments to the installer
