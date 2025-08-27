# Production Environment Variables

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "cybersecurity-threat-detection"
}

variable "owner" {
  description = "Owner of the resources"
  type        = string
  default     = "MLOpsTeam"
}

variable "source_type" {
  description = "Source type for CodePipeline (e.g., GITHUB, CODECOMMIT)"
  type        = string
  default     = "GITHUB"
  
}
variable "source_location" {
  description = "Location/URL of the source repository for CodePipeline"
  type        = string
  
}

variable "business_unit" {
  description = "Business unit responsible for this deployment"
  type        = string
  default     = "Security"
}

variable "contact_email" {
  description = "Contact email for this deployment"
  type        = string
}

# SageMaker Configuration
variable "endpoint_instance_type" {
  description = "SageMaker endpoint instance type for production"
  type        = string
  default     = "ml.m5.large"
  validation {
    condition = contains([
      "ml.m5.large",
      "ml.m5.xlarge", 
      "ml.m5.2xlarge",
      "ml.c5.xlarge",
      "ml.c5.2xlarge"
    ], var.endpoint_instance_type)
    error_message = "Instance type must be production-grade."
  }
}

variable "endpoint_instance_count" {
  description = "Number of instances for SageMaker endpoint"
  type        = number
  default     = 2
  validation {
    condition     = var.endpoint_instance_count >= 2 && var.endpoint_instance_count <= 10
    error_message = "Production should have 2-10 instances for high availability."
  }
}

# Auto Scaling Configuration
variable "enable_auto_scaling" {
  description = "Enable auto scaling for SageMaker endpoint"
  type        = bool
  default     = true
}

variable "auto_scaling_min_capacity" {
  description = "Minimum capacity for auto scaling"
  type        = number
  default     = 2
}

variable "auto_scaling_max_capacity" {
  description = "Maximum capacity for auto scaling"
  type        = number
  default     = 10
}

variable "auto_scaling_target_value" {
  description = "Target value for auto scaling metric"
  type        = number
  default     = 1000.0
}

# Storage Configuration
variable "lifecycle_transition_days" {
  description = "Days before transitioning to IA storage"
  type        = number
  default     = 30
}

variable "data_retention_days" {
  description = "Data retention period in days"
  type        = number
  default     = 365
}

variable "model_retention_days" {
  description = "Model retention period in days"
  type        = number
  default     = 730
}

variable "training_data_prefix" {
  description = "S3 prefix for training data"
  type        = string
  default     = "data/training/"
}

variable "model_artifacts_prefix" {
  description = "S3 prefix for model artifacts"
  type        = string
  default     = "models/"
}

# Security Configuration
variable "kms_key_id" {
  description = "KMS key ID for encryption (leave empty to create new)"
  type        = string
  default     = ""
}

variable "enable_vpc" {
  description = "Enable VPC for SageMaker and CodeBuild"
  type        = bool
  default     = true
}

variable "vpc_id" {
  description = "VPC ID for deployment"
  type        = string
  default     = ""
}

variable "subnet_ids" {
  description = "Private subnet IDs for VPC deployment"
  type        = list(string)
  default     = []
}

variable "security_group_ids" {
  description = "Security group IDs for VPC deployment"
  type        = list(string)
  default     = []
}

# CI/CD Configuration
variable "enable_cicd" {
  description = "Enable CI/CD pipeline for production"
  type        = bool
  default     = true
}

variable "github_repo" {
  description = "GitHub repository in format owner/repo"
  type        = string
  validation {
    condition     = can(regex("^[^/]+/[^/]+$", var.github_repo))
    error_message = "GitHub repository must be in format 'owner/repo'."
  }
}

variable "github_branch" {
  description = "GitHub branch for production"
  type        = string
  default     = "main"
}

variable "github_token" {
  description = "GitHub personal access token"
  type        = string
  sensitive   = true
}

variable "codebuild_compute_type" {
  description = "CodeBuild compute type for production"
  type        = string
  default     = "BUILD_GENERAL1_MEDIUM"
}

variable "build_timeout_minutes" {
  description = "Build timeout in minutes"
  type        = number
  default     = 120
}

variable "build_duration_threshold_minutes" {
  description = "Build duration alarm threshold in minutes"
  type        = number
  default     = 60
}

variable "pipeline_duration_threshold_minutes" {
  description = "Pipeline duration alarm threshold in minutes"
  type        = number
  default     = 180
}

# Monitoring Configuration
variable "sns_topic_arn" {
  description = "SNS topic ARN for notifications (leave empty to create new)"
  type        = string
  default     = ""
}

variable "alert_email" {
  description = "Email address for alerts"
  type        = string
  default     = ""
}

variable "log_retention_days" {
  description = "CloudWatch log retention period in days"
  type        = number
  default     = 30
}

# Alarm Thresholds
variable "invocation_error_threshold" {
  description = "Threshold for invocation errors alarm"
  type        = number
  default     = 5
}

variable "model_latency_threshold" {
  description = "Threshold for model latency alarm in milliseconds"
  type        = number
  default     = 1000
}

variable "low_invocations_threshold" {
  description = "Threshold for low invocations alarm"
  type        = number
  default     = 10
}