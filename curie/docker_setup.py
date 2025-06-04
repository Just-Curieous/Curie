import os
import sys
import subprocess
import platform
import logging

logger = logging.getLogger(__name__)

def is_docker_installed():
    """Check if Docker is installed on the system."""
    try:
        subprocess.run(['docker', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_docker():
    """Install Docker based on the operating system."""
    system = platform.system().lower()
    
    try:
        if system == 'linux':
            # Check if we're on Ubuntu/Debian
            if os.path.exists('/etc/debian_version'):
                # Update package list
                subprocess.run(['sudo', 'apt-get', 'update'], check=True)
                
                # Install prerequisites
                subprocess.run(['sudo', 'apt-get', 'install', '-y', 
                              'apt-transport-https', 
                              'ca-certificates', 
                              'curl', 
                              'software-properties-common'], check=True)
                
                # Add Docker's official GPG key
                subprocess.run(['curl', '-fsSL', 'https://download.docker.com/linux/ubuntu/gpg', 
                              '|', 'sudo', 'apt-key', 'add', '-'], shell=True, check=True)
                
                # Add Docker repository
                subprocess.run(['sudo', 'add-apt-repository', 
                              'deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable'], check=True)
                
                # Update package list again
                subprocess.run(['sudo', 'apt-get', 'update'], check=True)
                
                # Install Docker
                subprocess.run(['sudo', 'apt-get', 'install', '-y', 'docker-ce', 'docker-ce-cli', 'containerd.io'], check=True)
                
                # Add current user to docker group
                subprocess.run(['sudo', 'usermod', '-aG', 'docker', os.getenv('USER')], check=True)
                
                logger.info("Docker installed successfully. Please log out and log back in for the group changes to take effect.")
                return True
                
            else:
                logger.error("Automatic Docker installation is only supported on Ubuntu/Debian Linux systems.")
                return False
                
        elif system == 'darwin':  # macOS
            logger.error("Please install Docker Desktop for Mac from https://www.docker.com/products/docker-desktop")
            return False
            
        elif system == 'windows':
            logger.error("Please install Docker Desktop for Windows from https://www.docker.com/products/docker-desktop")
            return False
            
        else:
            logger.error(f"Unsupported operating system: {system}")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing Docker: {e}")
        return False

def ensure_docker_installed():
    """Ensure Docker is installed, install if necessary."""
    if not is_docker_installed():
        logger.info("Docker is not installed. Attempting to install...")
        if not install_docker():
            logger.error("Failed to install Docker. Please install Docker manually and try again.")
            sys.exit(1)
        logger.info("Docker installed successfully.")
    else:
        logger.info("Docker is already installed.") 