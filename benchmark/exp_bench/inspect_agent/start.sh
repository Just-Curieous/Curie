#!/bin/bash
# assumes it is run in container where the following env vars are set
# WORKSPACE_BASE, CODE_DIR, SUBMISSION_DIR, LOGS_DIR, AGENT_DIR

# Print commands and their arguments as they are executed
set -x
# bash start.sh /all/home/ubuntu/Benchmark-Construction/logs/neurips2024/MoE-Jetpack /all/home/ubuntu/Benchmark-Construction/logs/neurips2024/95262.json

# Check if env file argument is provided
if [ -z "$1" ]; then
    echo "Error: No env file specified"
    exit 1
fi

# Source the specified env file
source "$1"

# # move the original code directory '$2' to /home/code
# if [ -d "$1" ]; then
#   cp "$1" /home/code/
# else
#   echo "Error: Directory '$1' does not exist."
#   exit 1
# fi
# # move the original paper file '$2' to paper
# if [ -f "$2" ]; then
#   cp "$2" /home/paper/
# else
#   echo "Error: File '$2' does not exist."
#   exit 1
# fi

# cp instructions.txt /home/paper/

# Test Docker-in-Docker functionality
if [ -x "$(command -v docker)" ]; then
  docker --version
  # Skip to avoid get rate limited on docker pulling images
  # # Actually try to run a container
  # docker run --rm hello-world
  # # Show all containers that ran
  # echo "Listing all containers that ran, should include hello-world:"
  # docker ps -a
fi 2>&1 | tee $LOGS_DIR/docker.log

{  
  conda run -n agent --no-capture-output python start.py

  # Move agent logs to $LOGS_DIR/ directory
  if [ -d "$AGENT_DIR/logs" ]; then
    mv "$AGENT_DIR/logs"/* "$LOGS_DIR"/ 2> /dev/null || true
  fi

  # for debugging
  ls $WORKSPACE_BASE
  ls $CODE_DIR
  ls $SUBMISSION_DIR
  ls $LOGS_DIR
  ls $AGENT_DIR
} 2>&1 | tee $LOGS_DIR/agent.log
