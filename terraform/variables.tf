variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "cybersecurity-threat-detection"
}

variable "github_repo" {
  description = "GitHub repository URL"
  type        = string
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
}

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

variable "common_tags" {
  description = "Common tags to be applied to all resources"
  type        = map(string)
  default     = {}
}