variable "project_name" {
  description = "The name of the project"
  type        = string
}

variable "environment" {
  description = "The deployment environment (e.g., dev, prod)"
  type        = string
}

variable "service_role_arn" {
  description = "The ARN of the IAM role that CodeBuild will use"
  type        = string
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}

variable "artifacts_type" {
  description = "The type of artifacts CodeBuild should produce (e.g., S3)"
  type        = string
  default     = "NO_ARTIFACTS"
}

variable "artifacts_location" {
  description = "S3 bucket location for build artifacts"
  type        = string
  default     = null
}

variable "compute_type" {
  description = "Compute type for CodeBuild environment"
  type        = string
  default     = "BUILD_GENERAL1_SMALL"
}

variable "build_image" {
  description = "Docker image to use for CodeBuild environment"
  type        = string
}

variable "environment_type" {
  description = "Environment type for CodeBuild (e.g., LINUX_CONTAINER)"
  type        = string
  default     = "LINUX_CONTAINER"
}

variable "image_pull_credentials_type" {
  description = "How CodeBuild pulls the Docker image"
  type        = string
  default     = "CODEBUILD"
}

variable "privileged_mode" {
  description = "Whether to enable privileged mode (for Docker builds)"
  type        = bool
  default     = false
}

variable "environment_variables" {
  description = "Map of environment variables (PLAINTEXT) for the build"
  type        = map(string)
  default     = {}
}

variable "parameter_store_variables" {
  description = "Map of environment variables sourced from SSM Parameter Store"
  type        = map(string)
  default     = {}
}

variable "secrets_manager_variables" {
  description = "Map of environment variables sourced from AWS Secrets Manager"
  type        = map(string)
  default     = {}
}

variable "source_type" {
  description = "Source type for CodeBuild project (e.g., GITHUB, CODECOMMIT)"
  type        = string
}

variable "source_location" {
  description = "Location/URL of the source repository"
  type        = string
}

variable "buildspec_file" {
  description = "Path to the buildspec file"
  type        = string
}

variable "git_clone_depth" {
  description = "Depth for Git clone"
  type        = number
  default     = 1
}

variable "fetch_git_submodules" {
  description = "Whether to fetch Git submodules"
  type        = bool
  default     = false
}

variable "source_auth" {
  description = "Authentication configuration for source repository"
  type = object({
    type     = string
    resource = string
  })
  default = null
}

variable "secondary_sources" {
  description = "List of secondary sources"
  type = list(object({
    type                  = string
    location              = string
    source_identifier     = string
    buildspec             = string
    git_clone_depth       = number
    fetch_git_submodules  = bool
  }))
  default = []
}

variable "vpc_config" {
  description = "VPC configuration for CodeBuild"
  type = object({
    vpc_id             = string
    subnets            = list(string)
    security_group_ids = list(string)
  })
  default = null
}

variable "cache_config" {
  description = "Cache configuration for CodeBuild"
  type = object({
    type     = string
    location = string
    modes    = list(string)
  })
  default = null
}

variable "cloudwatch_logs_config" {
  description = "CloudWatch logs configuration"
  type = object({
    status      = string
    group_name  = string
    stream_name = string
  })
  default = null
}

variable "s3_logs_config" {
  description = "S3 logs configuration"
  type = object({
    status              = string
    location            = string
    encryption_disabled = bool
  })
  default = null
}

variable "build_timeout_minutes" {
  description = "Timeout for the build in minutes"
  type        = number
  default     = 60
}

variable "queued_timeout_minutes" {
  description = "Timeout for queued builds in minutes"
  type        = number
  default     = 480
}

variable "badge_enabled" {
  description = "Whether to enable the build badge"
  type        = bool
  default     = false
}

variable "concurrent_build_limit" {
  description = "Maximum number of concurrent builds"
  type        = number
  default     = 1
}

variable "create_cloudwatch_log_group" {
  description = "Whether to create a CloudWatch log group"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "Retention period for CloudWatch logs"
  type        = number
  default     = 30
}

variable "logs_kms_key_id" {
  description = "KMS key ID for CloudWatch logs encryption"
  type        = string
  default     = null
}

variable "create_webhook" {
  description = "Whether to create a CodeBuild webhook"
  type        = bool
  default     = false
}

variable "webhook_build_type" {
  description = "Build type for webhook (e.g., BUILD)"
  type        = string
  default     = "BUILD"
}

variable "webhook_filters" {
  description = "Filters for webhook events"
  type        = list(list(object({
    type    = string
    pattern = string
  })))
  default = []
}

variable "create_failure_alarm" {
  description = "Whether to create CloudWatch alarm for build failures"
  type        = bool
  default     = false
}

variable "create_duration_alarm" {
  description = "Whether to create CloudWatch alarm for build duration"
  type        = bool
  default     = false
}

variable "alarm_evaluation_periods" {
  description = "Number of evaluation periods for alarms"
  type        = number
  default     = 1
}

variable "alarm_period" {
  description = "Period for CloudWatch alarm in seconds"
  type        = number
  default     = 60
}

variable "failure_threshold" {
  description = "Threshold for build failure alarm"
  type        = number
  default     = 1
}

variable "duration_threshold_minutes" {
  description = "Threshold in minutes for build duration alarm"
  type        = number
  default     = 30
}

variable "alarm_actions" {
  description = "SNS topics or ARNs for alarm notifications"
  type        = list(string)
  default     = []
}
