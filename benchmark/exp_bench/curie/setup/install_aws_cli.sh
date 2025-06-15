# filename: install_aws_cli.sh
#!/bin/bash

# Install curl if not already installed
if ! command -v curl
then
    echo "Installing curl..."
    apt-get update && apt-get install -y curl
fi

# Install unzip if not already installed
if ! command -v unzip
then
    echo "Installing unzip..."
    apt-get update && apt-get install -y unzip
fi

# Install ssh if not already installed
if ! command -v ssh
then
    echo "Installing ssh..."
    apt-get update && apt-get install -y openssh-client
fi

if ! command -v less
then
    echo "Installing less..."
    apt-get update && apt-get install -y less
fi

# Download the AWS CLI bundle
echo "Downloading AWS CLI bundle..."
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"

# Unzip and install AWS CLI
echo "Unzipping AWS CLI package..."
unzip awscliv2.zip
echo "Installing AWS CLI..."
./aws/install