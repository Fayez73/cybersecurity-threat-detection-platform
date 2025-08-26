# CodeBuild Module Variables

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

# IAM Configuration
variable "service_role_arn" {
  description = "ARN of the CodeBuild service role"
  type        = string
}

# Build Configuration
variable "compute_type" {
  description = "Compute type for the build environment"
  type        = string
  default     = "BUILD_GENERAL1_MEDIUM"
  validation {
    condition = contains([
      "BUILD_GENERAL1_SMALL",
      "BUILD_GENERAL1_MEDIUM", 
      "BUILD_GENERAL1_LARGE",
      "BUILD_GENERAL1_2XLARGE"
    ], var.compute_type)
    error_message = "Compute type must be a valid CodeBuild compute type."
  }
}

variable "build_image" {
  description = "Docker image for the build environment"
  type        = string
  default     = "aws/codebuild/amazonlinux2-x86_64-standard:3.0"
}

variable "environment_type" {
  description = "Type of build environment"
  type        = string
  default     = "LINUX_CONTAINER"
}

variable "image_pull_credentials_type" {
  description = "Type of credentials CodeBuild uses to pull images"
  type        = string
  default     = "CODEBUILD"
}

variable "privileged_mode" {
  description