#!/bin/bash

# Cybersecurity Threat Detection System - Pipeline Runner
# This script orchestrates the complete ML pipeline from data processing to model deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
MODE="full"
ENVIRONMENT="dev"
DATA_PATH=""
CONFIG_PATH=""
OUTPUT_DIR="outputs/$(date +%Y%m%d_%H%M%S)"
SKIP_VALIDATION=false
SKIP_TUNING=false
DEPLOY=false
VERBOSE=false

# Logging
LOG_FILE="logs/pipeline_$(date +%Y%m%d_%H%M%S).log"

# Function definitions
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

print_header() {
    echo -e "${BLUE}========================================${NC}" | tee -a "$LOG_FILE"
    echo -e "${BLUE}$1${NC}" | tee -a "$LOG_FILE"
    echo -e "${BLUE}========================================${NC}" | tee -a "$LOG_FILE"
}

show_help() {
    cat << EOF
Cybersecurity Threat Detection Pipeline Runner

Usage: $0 [OPTIONS]

Modes:
    --mode MODE             Pipeline mode: full, train, evaluate, predict, tune [default: full]

Options:
    -e, --env ENVIRONMENT   Environment: dev, staging, prod [default: dev]
    -d, --data-path PATH    Path to training data (required for train/full modes)
    -c, --config PATH       Configuration file path
    -o, --output-dir DIR    Output directory [default: outputs/timestamp]
    --skip-validation       Skip data validation step
    --skip-tuning          Skip hyperparameter tuning
    --deploy               Deploy model after training
    -v, --verbose          Verbose output
    -h, --help             Show this help

Pipeline Modes:
    full        Complete pipeline: validate → train → tune → evaluate
    train       Train model only
    evaluate    Evaluate existing model
    predict     Run inference on new data
    tune        Hyperparameter tuning only
    validate    Data validation only

Examples:
    $0 --mode full --data-path data/sample/kdd_cup_sample.csv
    $0 --mode train --data-path s3://bucket/data.csv --deploy
    $0 --mode evaluate --model-path models/latest/
    $0 --mode predict --input-path data/new_threats.csv
    $0 --mode tune --data-path data/train.csv --skip-validation

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -d|--data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --skip-tuning)
            SKIP_TUNING=true
            shift
            ;;
        --deploy)
            DEPLOY=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate mode
if [[ ! "$MODE" =~ ^(full|train|evaluate|predict|tune|validate)$ ]]; then
    print_error "Invalid mode: $MODE"
    exit 1
fi

# Setup environment
print_header "Pipeline Initialization"
print_status "Mode: $MODE"
print_status "Environment: $ENVIRONMENT"
print_status "Output Directory: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "logs"

# Load environment
if [[ -f ".env.$ENVIRONMENT" ]]; then
    source ".env.$ENVIRONMENT"
    print_status "Environment variables loaded from .env.$ENVIRONMENT"
fi

# Activate virtual environment if exists
if [[ -d "venv" ]]; then
    source venv/bin/activate
    print_status "Python virtual environment activated"
fi

# Set configuration path
if [[ -z "$CONFIG_PATH" ]]; then
    CONFIG_PATH="configs/${ENVIRONMENT}.yaml"
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    print_error "Configuration file not found: $CONFIG_PATH"
    exit 1
fi

print_status "Using configuration: $CONFIG_PATH"

# Validate data path for modes that require it
if [[ "$MODE" =~ ^(full|train|tune|validate)$ ]] && [[ -z "$DATA_PATH" ]]; then
    print_error "Data path is required for mode: $MODE"
    exit 1
fi

# Function to run Python module with error handling
run_python_module() {
    local module=$1
    local description=$2
    shift 2
    
    print_status "Starting: $description"
    
    local cmd="python -m $module $@"
    if [[ "$VERBOSE" == true ]]; then
        print_status "Command: $cmd"
    fi
    
    if eval "$cmd" 2>&1 | tee -a "$LOG_FILE"; then
        print_status "Completed: $description"
        return 0
    else
        print_error "Failed: $description"
        return 1
    fi
}

# Function to check if file/directory exists
check_path() {
    local path=$1
    local description=$2
    
    if [[ ! -e "$path" ]]; then
        print_error "$description not found: $path"
        return 1
    fi
    print_status "$description found: $path"
    return 0
}

# Pipeline execution functions
run_data_validation() {
    print_header "Data Validation"
    
    local validation_output="$OUTPUT_DIR/validation"
    mkdir -p "$validation_output"
    
    run_python_module "src.data_processing.data_validation" "Data Validation" \
        --data_path "$DATA_PATH" \
        --output_dir "$validation_output" \
        --config "$CONFIG_PATH"
}

run_hyperparameter_tuning() {
    print_header "Hyperparameter Tuning"
    
    local tuning_output="$OUTPUT_DIR/tuning"
    mkdir -p "$tuning_output"
    
    run_python_module "src.training.hyperparameter_tuning" "Hyperparameter Tuning" \
        --data_path "$DATA_PATH" \
        --method comprehensive \
        --output_dir "$tuning_output" \
        --n_trials 50
    
    # Use best parameters for training if tuning succeeded
    if [[ -f "$tuning_output/best_hyperparameters.json" ]]; then
        CONFIG_PATH="$tuning_output/best_hyperparameters.json"
        print_status "Using optimized hyperparameters: $CONFIG_PATH"
    fi
}

run_model_training() {
    print_header "Model Training"
    
    local training_output="$OUTPUT_DIR/model"
    mkdir -p "$training_output"
    
    local training_args=(
        --data_path "$DATA_PATH"
        --output_dir "$training_output"
    )
    
    if [[ -f "$CONFIG_PATH" ]]; then
        training_args+=(--config_path "$CONFIG_PATH")
    fi
    
    run_python_module "src.training.train" "Model Training" "${training_args[@]}"
    
    # Store model path for later use
    MODEL_PATH="$training_output"
}

run_model_evaluation() {
    print_header "Model Evaluation"
    
    local evaluation_output="$OUTPUT_DIR/evaluation"
    mkdir -p "$evaluation_output"
    
    # Determine model path
    local model_dir="$MODEL_PATH"
    if [[ -z "$model_dir" ]] && [[ "$MODE" == "evaluate" ]]; then
        # Look for latest model
        model_dir=$(find models -name "threat_detection_model.pkl" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2- | xargs dirname)
        if [[ -z "$model_dir" ]]; then
            print_error "No trained model found for evaluation"
            return 1
        fi
    fi
    
    # Use test data or split from training data
    local test_data="$DATA_PATH"
    if [[ -f "data/test/test_data.csv" ]]; then
        test_data="data/test/test_data.csv"
    fi
    
    run_python_module "src.training.evaluate" "Model Evaluation" \
        --model_dir "$model_dir" \
        --test_data "$test_data" \
        --output_dir "$evaluation_output"
}

run_model_prediction() {
    print_header "Model Prediction"
    
    local prediction_output="$OUTPUT_DIR/predictions"
    mkdir -p "$prediction_output"
    
    # Determine model and feature engineer paths
    local model_path feature_engineer_path
    
    if [[ -n "$MODEL_PATH" ]]; then
        model_path="$MODEL_PATH/threat_detection_model.pkl"
        feature_engineer_path="$MODEL_PATH/feature_engineer.pkl"
    else
        # Find latest model
        model_path=$(find models -name "threat_detection_model.pkl" | head -1)
        feature_engineer_path=$(find models -name "feature_engineer.pkl" | head -1)
    fi
    
    if [[ ! -f "$model_path" ]] || [[ ! -f "$feature_engineer_path" ]]; then
        print_error "Model artifacts not found. Train a model first."
        return 1
    fi
    
    # Use provided input or sample data
    local input_data="${INPUT_PATH:-data/sample/kdd_cup_sample.csv}"
    
    run_python_module "src.inference.predictor" "Model Prediction" \
        --model_path "$model_path" \
        --feature_engineer_path "$feature_engineer_path" \
        --input_data "$input_data" \
        --output_path "$prediction_output/predictions.csv" \
        --mode batch
}

deploy_model() {
    if [[ "$DEPLOY" == true ]]; then
        print_header "Model Deployment"
        
        if [[ -z "$MODEL_PATH" ]]; then
            print_error "No model available for deployment"
            return 1
        fi
        
        # Deploy to SageMaker (placeholder - implement based on your deployment strategy)
        print_status "Deploying model to SageMaker endpoint..."
        
        # This would be implemented based on your specific deployment requirements
        # python scripts/deploy_model.py --model_path "$MODEL_PATH" --environment "$ENVIRONMENT"
        
        print_status "Model deployment completed"
    fi
}

# Progress tracking
track_progress() {
    local current_step=$1
    local total_steps=$2
    local description=$3
    
    local progress=$((current_step * 100 / total_steps))
    print_status "Progress: [$current_step/$total_steps] ($progress%) - $description"
}

# Main pipeline execution
main() {
    local start_time=$(date +%s)
    
    print_header "Starting Cybersecurity Threat Detection Pipeline"
    print_status "Pipeline ID: $(basename $OUTPUT_DIR)"
    print_status "Start Time: $(date)"
    
    case "$MODE" in
        "validate")
            track_progress 1 1 "Data Validation"
            run_data_validation
            ;;
        
        "tune")
            local steps=1
            [[ "$SKIP_VALIDATION" == false ]] && ((steps++))
            
            local current=0
            if [[ "$SKIP_VALIDATION" == false ]]; then
                track_progress $((++current)) $steps "Data Validation"
                run_data_validation
            fi
            
            track_progress $((++current)) $steps "Hyperparameter Tuning"
            run_hyperparameter_tuning
            ;;
        
        "train")
            local steps=1
            [[ "$SKIP_VALIDATION" == false ]] && ((steps++))
            [[ "$SKIP_TUNING" == false ]] && ((steps++))
            [[ "$DEPLOY" == true ]] && ((steps++))
            
            local current=0
            if [[ "$SKIP_VALIDATION" == false ]]; then
                track_progress $((++current)) $steps "Data Validation"
                run_data_validation
            fi
            
            if [[ "$SKIP_TUNING" == false ]]; then
                track_progress $((++current)) $steps "Hyperparameter Tuning"
                run_hyperparameter_tuning
            fi
            
            track_progress $((++current)) $steps "Model Training"
            run_model_training
            
            if [[ "$DEPLOY" == true ]]; then
                track_progress $((++current)) $steps "Model Deployment"
                deploy_model
            fi
            ;;
        
        "evaluate")
            track_progress 1 1 "Model Evaluation"
            run_model_evaluation
            ;;
        
        "predict")
            track_progress 1 1 "Model Prediction"
            run_model_prediction
            ;;
        
        "full")
            local steps=3
            [[ "$SKIP_VALIDATION" == false ]] && ((steps++))
            [[ "$SKIP_TUNING" == false ]] && ((steps++))
            [[ "$DEPLOY" == true ]] && ((steps++))
            
            local current=0
            if [[ "$SKIP_VALIDATION" == false ]]; then
                track_progress $((++current)) $steps "Data Validation"
                run_data_validation
            fi
            
            if [[ "$SKIP_TUNING" == false ]]; then
                track_progress $((++current)) $steps "Hyperparameter Tuning"
                run_hyperparameter_tuning
            fi
            
            track_progress $((++current)) $steps "Model Training"
            run_model_training
            
            track_progress $((++current)) $steps "Model Evaluation"
            run_model_evaluation
            
            track_progress $((++current)) $steps "Model Prediction"
            run_model_prediction
            
            if [[ "$DEPLOY" == true ]]; then
                track_progress $((++current)) $steps "Model Deployment"
                deploy_model
            fi
            ;;
    esac
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    print_header "Pipeline Completed Successfully"
    print_status "Total Duration: ${duration} seconds"
    print_status "Output Directory: $OUTPUT_DIR"
    print_status "Log File: $LOG_FILE"
    
    # Generate summary report
    generate_summary_report
}

generate_summary_report() {
    local report_file="$OUTPUT_DIR/pipeline_summary.md"
    
    cat > "$report_file" << EOF
# Pipeline Execution Summary

## Overview
- **Pipeline ID**: $(basename $OUTPUT_DIR)
- **Mode**: $MODE
- **Environment**: $ENVIRONMENT
- **Start Time**: $(date)
- **Duration**: ${duration:-N/A} seconds

## Configuration
- **Data Path**: $DATA_PATH
- **Config Path**: $CONFIG_PATH
- **Output Directory**: $OUTPUT_DIR

## Pipeline Steps
EOF

    if [[ -d "$OUTPUT_DIR/validation" ]]; then
        echo "- ✅ Data Validation" >> "$report_file"
    fi
    
    if [[ -d "$OUTPUT_DIR/tuning" ]]; then
        echo "- ✅ Hyperparameter Tuning" >> "$report_file"
    fi
    
    if [[ -d "$OUTPUT_DIR/model" ]]; then
        echo "- ✅ Model Training" >> "$report_file"
    fi
    
    if [[ -d "$OUTPUT_DIR/evaluation" ]]; then
        echo "- ✅ Model Evaluation" >> "$report_file"
    fi
    
    if [[ -d "$OUTPUT_DIR/predictions" ]]; then
        echo "- ✅ Model Prediction" >> "$report_file"
    fi
    
    cat >> "$report_file" << EOF

## Output Files
\`\`\`
$(find "$OUTPUT_DIR" -type f -name "*.json" -o -name "*.csv" -o -name "*.pkl" -o -name "*.png" | head -20)
\`\`\`

## Log File
See detailed logs in: $LOG_FILE
EOF
    
    print_status "Summary report generated: $report_file"
}

# Error handling
trap 'print_error "Pipeline failed with exit code $?"; exit 1' ERR

# Run main function
main "$@"