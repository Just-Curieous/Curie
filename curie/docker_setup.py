import os
import sys
import subprocess
import platform
import logging
import urllib.request
import tempfile
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

def is_docker_installed():
    """Check if Docker is installed on the system."""
    try:
        subprocess.run(['docker', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def is_homebrew_installed():
    """Check if Homebrew is installed on macOS."""
    try:
        subprocess.run(['brew', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_homebrew():
    """Install Homebrew on macOS."""
    logger.info("Installing Homebrew...")
    try:
        # Download and run the Homebrew installation script
        install_script = '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        subprocess.run(install_script, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install Homebrew: {e}")
        return False

def is_chocolatey_installed():
    """Check if Chocolatey is installed on Windows."""
    try:
        subprocess.run(['choco', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_chocolatey():
    """Install Chocolatey on Windows."""
    logger.info("Installing Chocolatey...")
    try:
        # PowerShell command to install Chocolatey
        powershell_cmd = [
            'powershell.exe', '-NoProfile', '-InputFormat', 'None', '-ExecutionPolicy', 'Bypass',
            '-Command', 'Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString(\'https://community.chocolatey.org/install.ps1\'))'
        ]
        subprocess.run(powershell_cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install Chocolatey: {e}")
        return False

def download_file(url, destination):
    """Download a file from URL to destination."""
    try:
        urllib.request.urlretrieve(url, destination)
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False

def install_docker_macos():
    """Install Docker on macOS using multiple methods."""
    logger.info("Installing Docker on macOS...")
    
    # Method 1: Try using Homebrew (preferred)
    if is_homebrew_installed() or install_homebrew():
        try:
            logger.info("Installing Docker using Homebrew...")
            subprocess.run(['brew', 'install', '--cask', 'docker'], check=True)
            logger.info("Docker installed via Homebrew. Please start Docker Desktop from Applications.")
            return True
        except subprocess.CalledProcessError as e:
            logger.warning(f"Homebrew installation failed: {e}")
    
    # Method 2: Download Docker Desktop directly
    logger.info("Attempting direct download of Docker Desktop...")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            docker_dmg = os.path.join(temp_dir, "Docker.dmg")
            docker_url = "https://desktop.docker.com/mac/main/amd64/Docker.dmg"
            
            # Detect Apple Silicon
            if platform.machine() == 'arm64':
                docker_url = "https://desktop.docker.com/mac/main/arm64/Docker.dmg"
            
            logger.info(f"Downloading Docker Desktop from {docker_url}...")
            if download_file(docker_url, docker_dmg):
                # Mount the DMG
                logger.info("Mounting Docker DMG...")
                mount_result = subprocess.run(['hdiutil', 'attach', docker_dmg], 
                                            capture_output=True, text=True)
                if mount_result.returncode == 0:
                    # Copy Docker app to Applications
                    logger.info("Installing Docker to Applications folder...")
                    subprocess.run(['cp', '-R', '/Volumes/Docker/Docker.app', '/Applications/'], check=True)
                    
                    # Unmount the DMG
                    subprocess.run(['hdiutil', 'detach', '/Volumes/Docker'], check=False)
                    
                    logger.info("Docker Desktop installed successfully. Please launch it from Applications.")
                    return True
    except Exception as e:
        logger.error(f"Direct download installation failed: {e}")
    
    return False

def install_docker_windows():
    """Install Docker on Windows using multiple methods."""
    logger.info("Installing Docker on Windows...")
    
    # Method 1: Try using Chocolatey (preferred)
    if is_chocolatey_installed() or install_chocolatey():
        try:
            logger.info("Installing Docker using Chocolatey...")
            subprocess.run(['choco', 'install', 'docker-desktop', '-y'], check=True)
            logger.info("Docker installed via Chocolatey. Please restart your system.")
            return True
        except subprocess.CalledProcessError as e:
            logger.warning(f"Chocolatey installation failed: {e}")
    
    # Method 2: Try using winget (Windows Package Manager)
    try:
        logger.info("Attempting to install Docker using winget...")
        subprocess.run(['winget', 'install', 'Docker.DockerDesktop'], check=True)
        logger.info("Docker installed via winget. Please restart your system.")
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(f"Winget installation failed: {e}")
    
    # Method 3: Download Docker Desktop directly
    logger.info("Attempting direct download of Docker Desktop...")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            docker_exe = os.path.join(temp_dir, "DockerDesktopInstaller.exe")
            docker_url = "https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe"
            
            logger.info(f"Downloading Docker Desktop from {docker_url}...")
            if download_file(docker_url, docker_exe):
                logger.info("Running Docker Desktop installer...")
                # Run the installer silently
                subprocess.run([docker_exe, 'install', '--quiet'], check=True)
                logger.info("Docker Desktop installed successfully. Please restart your system.")
                return True
    except Exception as e:
        logger.error(f"Direct download installation failed: {e}")
    
    return False

def install_docker_linux():
    """Install Docker on Linux (existing implementation)."""
    logger.info("Installing Docker on Linux...")
    
    # Check if we're on Ubuntu/Debian
    if os.path.exists('/etc/debian_version'):
        try:
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
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing Docker on Linux: {e}")
            return False
    else:
        logger.error("Automatic Docker installation is only supported on Ubuntu/Debian Linux systems.")
        return False

def install_docker():
    """Install Docker based on the operating system."""
    system = platform.system().lower()
    
    if system == 'linux':
        return install_docker_linux()
    elif system == 'darwin':  # macOS
        return install_docker_macos()
    elif system == 'windows':
        return install_docker_windows()
    else:
        logger.error(f"Unsupported operating system: {system}")
        return False

def wait_for_docker():
    """Wait for Docker daemon to be ready."""
    import time
    max_attempts = 30
    attempt = 0
    
    while attempt < max_attempts:
        try:
            subprocess.run(['docker', 'info'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            logger.info("Docker daemon is ready.")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.info(f"Waiting for Docker daemon... (attempt {attempt + 1}/{max_attempts})")
            time.sleep(10)
            attempt += 1
    
    logger.warning("Docker daemon is not responding. You may need to start Docker Desktop manually.")
    return False

def ensure_docker_installed():
    """Ensure Docker is installed, install if necessary."""
    if not is_docker_installed():
        logger.info("Docker is not installed. Attempting to install...")
        if not install_docker():
            logger.error("Failed to install Docker. Please install Docker manually and try again.")
            sys.exit(1)
        
        # For macOS and Windows, Docker Desktop needs to be started
        system = platform.system().lower()
        if system in ['darwin', 'windows']:
            logger.info("Docker Desktop has been installed. Please start Docker Desktop and then run this script again.")
            logger.info("On macOS: Open Applications folder and launch Docker")
            logger.info("On Windows: Look for Docker Desktop in your Start menu")
            return False
        
        logger.info("Docker installed successfully.")
    else:
        logger.info("Docker is already installed.")
    
    # Check if Docker daemon is running
    try:
        subprocess.run(['docker', 'info'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        logger.info("Docker daemon is running.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        system = platform.system().lower()
        if system in ['darwin', 'windows']:
            logger.info("Docker is installed but not running. Please start Docker Desktop.")
            return False
        else:
            logger.info("Starting Docker daemon...")
            try:
                subprocess.run(['sudo', 'systemctl', 'start', 'docker'], check=True)
                subprocess.run(['sudo', 'systemctl', 'enable', 'docker'], check=True)
                return wait_for_docker()
            except subprocess.CalledProcessError:
                logger.error("Failed to start Docker daemon.")
                return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    ensure_docker_installed()