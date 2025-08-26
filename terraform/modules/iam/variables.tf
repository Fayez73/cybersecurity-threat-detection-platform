# IAM Module Variables

variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "tags" {
  description = "Tags to be applied to all resources"
  type        = map(string)
  default     = {}
}

# S3 Configuration
variable "s3_bucket_arns" {
  description = "List of S3 bucket ARNs for SageMaker access"
  type        = list(string)
  default     = []
}

variable "pipeline_s3_bucket_arn" {
  description = "S3 bucket ARN for pipeline artifacts"
  type        = string
  default     = null
}

# VPC Configuration
variable "enable_vpc" {
  description = "Enable VPC access for SageMaker"
  type        = bool
  default     = false
}

# Role Creation Flags
variable "create_codebuild_role" {
  description = "Create IAM role for CodeBuild"
  type        = bool
  default     = false
}

variable "create_codepipeline_role" {
  description = "Create IAM role for CodePipeline"
  type        = bool
  default     = false
}

# Additional Permissions
variable "additional_sagemaker_permissions" {
  description = "Additional IAM permissions for SageMaker role"
  type        = list(string)
  default     = []
}

variable "additional_codebuild_permissions" {
  description = "Additional IAM permissions for CodeBuild role"
  type        = list(string)
  default     = []
}