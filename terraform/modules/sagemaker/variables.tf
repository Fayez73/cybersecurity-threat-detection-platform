variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "cybersecurity-threat-detection"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "common_tags" {
  description = "Common tags to be applied to all resources"
  type        = map(string)
  default     = {}
}

# S3 Configuration

# SageMaker Configuration
variable "sagemaker_instance_type" {
  description = "SageMaker instance type for training"
  type        = string
  default     = "ml.m5.xlarge"
}

variable "endpoint_instance_type" {
  description = "SageMaker instance type for endpoint"
  type        = string
  default     = "ml.t2.medium"
}

variable "endpoint_instance_count" {
  description = "Number of instances for SageMaker endpoint"
  type        = number
  default     = 1
}

variable "auto_scaling_enabled" {
  description = "Enable auto scaling for SageMaker endpoint"
  type        = bool
  default     = false
}

variable "auto_scaling_max_capacity" {
  description = "Maximum capacity for auto scaling"
  type        = number
  default     = 3
}

variable "auto_scaling_min_capacity" {
  description = "Minimum capacity for auto scaling"
  type        = number
  default     = 1
}

# CI/CD Configuration
variable "enable_cicd" {
  description = "Enable CI/CD pipeline"
  type        = bool
  default     = true
}

variable "github_repo" {
  description = "GitHub repository (owner/repo)"
  type        = string
  default     = ""
}

variable "github_branch" {
  description = "GitHub branch"
  type        = string
  default     = "main"
}

variable "github_token" {
  description = "GitHub personal access token"
  type        = string
  sensitive   = true
  default     = ""
}

variable "codebuild_image" {
  description = "CodeBuild image"
  type        = string
  default     = "aws/codebuild/amazonlinux2-x86_64-standard:3.0"
}

variable "codebuild_compute_type" {
  description = "CodeBuild compute type"
  type        = string
  default     = "BUILD_GENERAL1_MEDIUM"
}

# Security Configuration
variable "enable_vpc_endpoints" {
  description = "Enable VPC endpoints for S3 and SageMaker"
  type        = bool
  default     = false
}

variable "vpc_id" {
  description = "VPC ID for VPC endpoints"
  type        = string
  default     = ""
}

variable "subnet_ids" {
  description = "Subnet IDs for VPC endpoints"
  type        = list(string)
  default     = []
}

variable "enable_encryption" {
  description = "Enable encryption for S3 buckets"
  type        = bool
  default     = true
}

variable "kms_key_id" {
  description = "KMS key ID for encryption"
  type        = string
  default     = ""
}

# Monitoring Configuration
variable "enable_cloudwatch_alarms" {
  description = "Enable CloudWatch alarms"
  type        = bool
  default     = true
}

variable "sns_topic_arn" {
  description = "SNS topic ARN for notifications"
  type        = string
  default     = ""
}

# Data Configuration
variable "training_data_prefix" {
  description = "S3 prefix for training data"
  type        = string
  default     = "data/training"
}

variable "model_artifacts_prefix" {
  description = "S3 prefix for model artifacts"
  type        = string
  default     = "models"
}

variable "data_retention_days" {
  description = "Data retention period in days"
  type        = number
  default     = 90
}