# CodePipeline Module Variables

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
  description = "ARN of the CodePipeline service role"
  type        = string
}

# Artifact Store Configuration
variable "artifact_store_bucket" {
  description = "S3 bucket for pipeline artifacts"
  type        = string
}

variable "artifact_store_kms_key_id" {
  description = "KMS key ID for artifact store encryption"
  type        = string
  default     = null
}

# Source Actions Configuration
variable "source_actions" {
  description = "List of source actions"
  type = list(object({
    name             = string
    owner            = string
    provider         = string
    version          = string
    output_artifacts = list(string)
    run_order        = optional(number)
    input_artifacts  = optional(list(string))
    configuration    = map(string)
    namespace        = optional(string)
    region           = optional(string)
  }))
}

# Build Actions Configuration
variable "build_actions" {
  description = "List of build actions"
  type = list(object({
    name             = string
    owner            = string
    provider         = string
    version          = string
    input_artifacts  = list(string)
    output_artifacts = optional(list(string))
    run_order        = optional(number)
    configuration    = map(string)
    namespace        = optional(string)
    region           = optional(string)
  }))
  default = null
}

# Test Actions Configuration
variable "test_actions" {
  description = "List of test actions"
  type = list(object({
    name             = string
    category         = string
    owner            = string
    provider         = string
    version          = string
    input_artifacts  = optional(list(string))
    output_artifacts = optional(list(string))
    run_order        = optional(number)
    configuration    = map(string)
    namespace        = optional(string)
    region           = optional(string)
  }))
  default = null
}

# Deploy Actions Configuration
variable "deploy_actions" {
  description = "List of deploy actions"
  type = list(object({
    name             = string
    category         = string
    owner            = string
    provider         = string
    version          = string
    input_artifacts  = optional(list(string))
    output_artifacts = optional(list(string))
    run_order        = optional(number)
    configuration    = map(string)
    namespace        = optional(string)
    region           = optional(string)
  }))
  default = null
}

# Custom Stages Configuration
variable "custom_stages" {
  description = "List of custom stages with their actions"
  type = list(object({
    name = string
    actions = list(object({
      name             = string
      category         = string
      owner            = string
      provider         = string
      version          = string
      input_artifacts  = optional(list(string))
      output_artifacts = optional(list(string))
      run_order        = optional(number)
      configuration    = map(string)
      namespace        = optional(string)
      region           = optional(string)
    }))
  }))
  default = []
}

# Event Configuration
variable "create_pipeline_events" {
  description = "Create CloudWatch events for pipeline state changes"
  type        = bool
  default     = false
}

variable "event_target_arn" {
  description = "ARN of the event target (SNS, Lambda, etc.)"
  type        = string
  default     = null
}

variable "event_input_transformer" {
  description = "Input transformer for CloudWatch events"
  type = object({
    input_paths    = map(string)
    input_template = string
  })
  default = null
}

# CloudWatch Alarms Configuration
variable "create_failure_alarm" {
  description = "Create CloudWatch alarm for pipeline failures"
  type        = bool
  default     = true
}

variable "create_duration_alarm" {
  description = "Create CloudWatch alarm for pipeline duration"
  type        = bool
  default     = true
}

variable "alarm_actions" {
  description = "List of alarm action ARNs"
  type        = list(string)
  default     = []
}

variable "alarm_period" {
  description = "Period for CloudWatch alarms in seconds"
  type        = number
  default     = 300
}

variable "alarm_evaluation_periods" {
  description = "Number of evaluation periods for alarms"
  type        = number
  default     = 1
}

variable "failure_threshold" {
  description = "Threshold for pipeline failure alarm"
  type        = number
  default     = 0
}

variable "duration_threshold_minutes" {
  description = "Threshold for pipeline duration alarm in minutes"
  type        = number
  default     = 60
}