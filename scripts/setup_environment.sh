#!/bin/bash

# Cybersecurity Threat Detection System - Environment Setup Script
# This script sets up the development environment and prepares the system for deployment

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="dev"
PYTHON_VERSION="3.8"
SKIP_DEPS=false
SKIP_DATA=false
AWS_REGION="us-east-1"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Function to display help
show_help() {
    cat << EOF
Cybersecurity Threat Detection System - Environment Setup

Usage: $0 [OPTIONS]

Options:
    -e, --env ENVIRONMENT       Set environment (dev, staging, prod) [default: dev]
    -p, --python VERSION        Python version to use [default: 3.8]
    -r, --region REGION         AWS region [default: us-east-1]
    --skip-deps                 Skip dependency installation
    --skip-data                 Skip sample data setup
    -h, --help                  Show this help message

Examples:
    $0                          # Setup development environment
    $0 --env prod               # Setup production environment
    $0 --skip-deps              # Skip dependency installation
    $0 --env staging --region us-west-2  # Custom environment and region

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -p|--python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        -r|--region)
            AWS_REGION="$2"
            shift 2
            ;;
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --skip-data)
            SKIP_DATA=true
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

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|prod)$ ]]; then
    print_error "Invalid environment: $ENVIRONMENT. Must be one of: dev, staging, prod"
    exit 1
fi

print_header "Cybersecurity Threat Detection System Setup"
print_status "Environment: $ENVIRONMENT"
print_status "Python Version: $PYTHON_VERSION"
print_status "AWS Region: $AWS_REGION"

# Check prerequisites
print_header "Checking Prerequisites"

# Check Python version
if ! command -v python$PYTHON_VERSION &> /dev/null; then
    print_error "Python $PYTHON_VERSION is not installed"
    print_status "Please install Python $PYTHON_VERSION first"
    exit 1
fi
print_status "Python $PYTHON_VERSION: $(python$PYTHON_VERSION --version)"

# Check pip
if ! command -v pip &> /dev/null; then
    print_error "pip is not installed"
    exit 1
fi
print_status "pip: $(pip --version)"

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    print_warning "AWS CLI not found. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew install awscli
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
        unzip awscliv2.zip
        sudo ./aws/install
        rm -rf aws awscliv2.zip
    fi
fi
print_status "AWS CLI: $(aws --version)"

# Check Terraform (optional)
if command -v terraform &> /dev/null; then
    print_status "Terraform: $(terraform version -json | jq -r '.terraform_version')"
else
    print_warning "Terraform not found. Infrastructure deployment will not be available."
fi

# Setup Python virtual environment
print_header "Setting Up Python Environment"

if [[ -d "venv" ]]; then
    print_warning "Virtual environment already exists. Removing..."
    rm -rf venv
fi

print_status "Creating virtual environment with Python $PYTHON_VERSION"
python$PYTHON_VERSION -m venv venv

# Activate virtual environment
source venv/bin/activate
print_status "Virtual environment activated"

# Upgrade pip
print_status "Upgrading pip"
pip install --upgrade pip

# Install dependencies
if [[ "$SKIP_DEPS" == false ]]; then
    print_header "Installing Dependencies"
    
    if [[ -f "requirements.txt" ]]; then
        print_status "Installing Python dependencies from requirements.txt"
        pip install -r requirements.txt
    else
        print_error "requirements.txt not found"
        exit 1
    fi
    
    # Install development dependencies for dev environment
    if [[ "$ENVIRONMENT" == "dev" ]]; then
        print_status "Installing development dependencies"
        pip install jupyter notebook ipywidgets
    fi
    
    print_status "Dependencies installed successfully"
else
    print_warning "Skipping dependency installation"
fi

# Create directory structure
print_header "Creating Directory Structure"

directories=(
    "data/raw"
    "data/processed" 
    "data/sample"
    "models"
    "logs"
    "notebooks"
    "tests"
    "configs"
    "outputs"
)

for dir in "${directories[@]}"; do
    if [[ ! -d "$dir" ]]; then
        mkdir -p "$dir"
        print_status "Created directory: $dir"
    fi
done

# Setup configuration files
print_header "Setting Up Configuration"

# Create environment-specific config
config_file="configs/${ENVIRONMENT}.yaml"
cat > "$config_file" << EOF
# Configuration for $ENVIRONMENT environment
model:
  objective: 'binary:logistic'
  max_depth: 6
  learning_rate: 0.1
  n_estimators: 100
  subsample: 0.8
  colsample_bytree: 0.8
  random_state: 42
  eval_metric: 'auc'
  early_stopping_rounds: 10

data:
  test_size: 0.2
  val_size: 0.1
  random_state: 42
  normalize_features: true
  remove_outliers: true
  outlier_threshold: 3.0
  categorical_encoding: 'onehot'

training:
  batch_size: 1000
  max_epochs: 1000
  validation_split: 0.2
  cross_validation_folds: 5
  hyperparameter_tuning: true
  tuning_trials: 50

inference:
  batch_size: 1000
  output_format: 'json'
  include_probabilities: true
  threshold: 0.5

aws:
  region: $AWS_REGION
  s3_bucket: cyber-threat-detection-${ENVIRONMENT}
  instance_type: ml.m5.large
  training_instance_type: ml.m5.xlarge
  endpoint_name: threat-detection-endpoint-${ENVIRONMENT}
EOF

print_status "Created configuration file: $config_file"

# Environment variables setup
env_file=".env.${ENVIRONMENT}"
cat > "$env_file" << EOF
# Environment variables for $ENVIRONMENT
export PYTHONPATH=\$PYTHONPATH:$(pwd)
export AWS_DEFAULT_REGION=$AWS_REGION
export ENVIRONMENT=$ENVIRONMENT
export LOG_LEVEL=INFO
export MODEL_DIR=models
export DATA_DIR=data
export CONFIG_FILE=configs/${ENVIRONMENT}.yaml
EOF

print_status "Created environment file: $env_file"

# Setup logging configuration
log_config="configs/logging.yaml"
cat > "$log_config" << EOF
version: 1
formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  detailed:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: default
    stream: ext://sys.stdout

  file:
    class: logging.FileHandler
    level: DEBUG
    formatter: detailed
    filename: logs/application.log
    mode: a

  error_file:
    class: logging.FileHandler
    level: ERROR
    formatter: detailed
    filename: logs/errors.log
    mode: a

root:
  level: INFO
  handlers: [console, file, error_file]

loggers:
  src:
    level: DEBUG
    handlers: [console, file]
    propagate: false
EOF

print_status "Created logging configuration: $log_config"

# Setup sample data
if [[ "$SKIP_DATA" == false ]]; then
    print_header "Setting Up Sample Data"
    
    # Generate sample data if not exists
    if [[ ! -f "data/sample/kdd_cup_sample.csv" ]]; then
        print_status "Generating sample threat detection data"
        python -c "
from src.data_processing.data_loader import DataLoader
from src.utils.config import Config

config = Config('$config_file')
loader = DataLoader(config)
sample_data = loader.create_sample_threat_data(n_samples=5000)
sample_data.to_csv('data/sample/kdd_cup_sample.csv', index=False)
print('Sample data created successfully')
"
    fi
    
    print_status "Sample data is ready"
else
    print_warning "Skipping sample data setup"
fi

# AWS Configuration
print_header "AWS Configuration"

# Check AWS credentials
if aws sts get-caller-identity &> /dev/null; then
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    print_status "AWS credentials configured for account: $ACCOUNT_ID"
    
    # Create S3 bucket if it doesn't exist (dev environment only)
    if [[ "$ENVIRONMENT" == "dev" ]]; then
        bucket_name="cyber-threat-detection-${ENVIRONMENT}-${ACCOUNT_ID}"
        
        if ! aws s3 ls "s3://$bucket_name" &> /dev/null; then
            print_status "Creating S3 bucket: $bucket_name"
            aws s3 mb "s3://$bucket_name" --region "$AWS_REGION"
        else
            print_status "S3 bucket already exists: $bucket_name"
        fi
        
        # Update config with actual bucket name
        sed -i.bak "s/cyber-threat-detection-${ENVIRONMENT}/$bucket_name/g" "$config_file"
        rm "$config_file.bak"
    fi
else
    print_warning "AWS credentials not configured. Run 'aws configure' to set up AWS access."
fi

# Setup git hooks (development environment)
if [[ "$ENVIRONMENT" == "dev" && -d ".git" ]]; then
    print_header "Setting Up Git Hooks"
    
    # Pre-commit hook for code formatting
    cat > ".git/hooks/pre-commit" << 'EOF'
#!/bin/bash
# Format Python code with black
if command -v black &> /dev/null; then
    black --check src/ || (echo "Run 'black src/' to format code" && exit 1)
fi

# Sort imports with isort
if command -v isort &> /dev/null; then
    isort --check-only src/ || (echo "Run 'isort src/' to sort imports" && exit 1)
fi

# Run flake8 for linting
if command -v flake8 &> /dev/null; then
    flake8 src/ || exit 1
fi
EOF
    
    chmod +x ".git/hooks/pre-commit"
    print_status "Git pre-commit hook installed"
fi

# Create useful scripts
print_header "Creating Utility Scripts"

# Quick start script
cat > "quick_start.sh" << EOF
#!/bin/bash
# Quick start script for threat detection system

echo "Loading environment..."
source venv/bin/activate
source .env.$ENVIRONMENT

echo "Starting Jupyter notebook for exploration..."
jupyter notebook notebooks/
EOF
chmod +x quick_start.sh

# Test script
cat > "run_tests.sh" << EOF
#!/bin/bash
# Run test suite

source venv/bin/activate
source .env.$ENVIRONMENT

echo "Running tests..."
python -m pytest tests/ -v --cov=src --cov-report=html

echo "Test results saved to htmlcov/index.html"
EOF
chmod +x run_tests.sh

print_status "Utility scripts created"

# Final summary
print_header "Setup Complete!"

echo -e "${GREEN}✓${NC} Python virtual environment created and activated"
echo -e "${GREEN}✓${NC} Dependencies installed"
echo -e "${GREEN}✓${NC} Directory structure created"
echo -e "${GREEN}✓${NC} Configuration files generated"
echo -e "${GREEN}✓${NC} Environment: $ENVIRONMENT"

echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "1. Activate the environment: source venv/bin/activate && source .env.$ENVIRONMENT"
echo "2. Configure AWS credentials: aws configure (if not done already)"
echo "3. Train your first model: python -m src.training.train --data_path data/sample/kdd_cup_sample.csv"
echo "4. Start exploring: ./quick_start.sh"

echo ""
echo -e "${YELLOW}Important Files:${NC}"
echo "• Configuration: $config_file"
echo "• Environment variables: $env_file"
echo "• Sample data: data/sample/kdd_cup_sample.csv"
echo "• Logs: logs/"

print_status "Setup completed successfully! 🎉"