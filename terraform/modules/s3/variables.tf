# S3 Module Variables

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

# Bucket Configuration
variable "enable_versioning" {
  description = "Enable S3 bucket versioning"
  type        = bool
  default     = true
}

variable "enable_encryption" {
  description = "Enable S3 bucket encryption"
  type        = bool
  default     = true
}

variable "kms_key_id" {
  description = "KMS key ID for S3 bucket encryption"
  type        = string
  default     = null
}

variable "enable_lifecycle" {
  description = "Enable S3 lifecycle rules"
  type        = bool
  default     = true
}

variable "lifecycle_transition_days" {
  description = "Number of days before transitioning to IA storage class"
  type        = number
  default     = 30
}

variable "data_retention_days" {
  description = "Number of days to retain training data"
  type        = number
  default     = 365
}

variable "model_retention_days" {
  description = "Number of days to retain model artifacts"
  type        = number
  default     = 730
}

# Data Configuration
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

# Pipeline Configuration
variable "create_pipeline_bucket" {
  description = "Create S3 bucket for pipeline artifacts"
  type        = bool
  default     = false
}

# Notification Configuration
variable "sns_topic_arn" {
  description = "SNS topic ARN for S3 notifications"
  type        = string
  default     = null
}