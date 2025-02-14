#!/bin/bash

QUESTION_DIR="benchmark/experimentation_bench/llm_reasoning"
CONFIG_PATH="curie/configs/llm_reasoning_config.json"
LOG_FILE="experiment_results.log"

# Create/clear log file
echo "Starting experiments at $(date)" > $LOG_FILE

# Find and process all q* files
for question_file in $QUESTION_DIR/q*.txt; do
    echo "Processing: $question_file"
    echo "=== Running $question_file ===" >> $LOG_FILE
    echo "Start time: $(date)" >> $LOG_FILE
    
    # Run command and capture real-time output
    python3 -m curie.main \
        --iterations 1 \
        --question_file "$question_file" \
        --task_config "$CONFIG_PATH" 2>&1 | tee -a $LOG_FILE
    
    echo "End time: $(date)" >> $LOG_FILE
    echo "----------------------------------------" >> $LOG_FILE
done

echo "All experiments completed at $(date)" >> $LOG_FILE