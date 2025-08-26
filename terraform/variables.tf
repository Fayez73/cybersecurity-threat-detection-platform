# SageMaker Module Variables

variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
}

variable "tags" {
  description = "Tags to be applied to all resources"
  type        = map(string)
  default     = {}
}

# IAM Configuration
variable "sagemaker_execution_role_arn" {
  description = "ARN of the SageMaker execution role"
  type        = string
}

# Model Configuration
variable "training_image_uri" {
  description = "URI of the training image"
  type        = string
  default     = "246618743249.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.5-1"
}

variable "model_data_url" {
  description = "S3 URL of the model artifacts"
  type        = string
}

variable "inference_script_name" {
  description = "Name of the inference script"
  type        = string
  default     = "inference.py"
}

variable "model_environment_variables" {
  description = "Environment variables for the model container"
  type        = map(string)
  default     = {}
}

variable "log_level" {
  description = "Log level for SageMaker container"
  type        = string
  default     = "20"
}

# Endpoint Configuration
variable "endpoint_instance_type" {
  description = "Instance type for the SageMaker endpoint"
  type        = string
  default     = "ml.t2.medium"
}

variable "endpoint_instance_count" {
  description = "Number of instances for the SageMaker endpoint"
  type        = number
  default     = 1
}

# VPC Configuration
variable "vpc_config" {
  description = "VPC configuration for SageMaker"
  type = object({
    subnets            = list(string)
    security_group_ids = list(string)
  })
  default = null
}

# Data Capture Configuration
variable "data_capture_config" {
  description = "Data capture configuration for model monitoring"
  type = object({
    enable_capture              = bool
    initial_sampling_percentage = number
    destination_s3_uri         = string
    capture_options            = list(string)
    capture_content_type_header = optional(object({
      csv_content_types  = list(string)
      json_content_types = list(string)
    }))
  })
  default = null
}

# Encryption
variable "kms_key_id" {
  description = "KMS key ID for encryption"
  type        = string
  default     = null
}

# Blue/Green Deployment
variable "blue_green_update_policy" {
  description = "Blue/green update policy configuration"
  type = object({
    traffic_routing_configuration = object({
      type                     = string
      wait_interval_in_seconds = number
      canary_size = optional(object({
        type  = string
        value = number
      }))
    })
    termination_wait_in_seconds          = number
    maximum_execution_timeout_in_seconds = number
  })
  default = null
}

variable "auto_rollback_alarms" {
  description = "List of CloudWatch alarm names for auto rollback"
  type        = list(string)
  default     = []
}

# Auto Scaling Configuration
variable "enable_auto_scaling" {
  description = "Enable auto scaling for the SageMaker endpoint"
  type        = bool
  default     = false
}

variable "auto_scaling_min_capacity" {
  description = "Minimum capacity for auto scaling"
  type        = number
  default     = 1
}

variable "auto_scaling_max_capacity" {
  description = "Maximum capacity for auto scaling"
  type        = number
  default     = 10
}

variable "auto_scaling_metric_type" {
  description = "Metric type for auto scaling"
  type        = string
  default     = "SageMakerVariantInvocationsPerInstance"
}

variable "auto_scaling_target_value" {
  description = "Target value for auto scaling metric"
  type        = number
  default     = 1000.0
}

variable "auto_scaling_scale_in_cooldown" {
  description = "Scale in cooldown period in seconds"
  type        = number
  default     = 300
}

variable "auto_scaling_scale_out_cooldown" {
  description = "Scale out cooldown period in seconds"
  type        = number
  default     = 60
}

# CloudWatch Alarms
variable "create_cloudwatch_alarms" {
  description = "Create CloudWatch alarms for monitoring"
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
  default     = 2
}

variable "invocation_error_threshold" {
  description = "Threshold for invocation errors alarm"
  type        = number
  default     = 10
}

variable "model_latency_threshold" {
  description = "Threshold for model latency alarm in milliseconds"
  type        = number
  default     = 2000
}

variable "low_invocations_threshold" {
  description = "Threshold for low invocations alarm"
  type        = number
  default     = 1
}