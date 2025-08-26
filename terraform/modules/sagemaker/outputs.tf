# SageMaker Module Outputs

output "model_name" {
  description = "Name of the SageMaker model"
  value       = aws_sagemaker_model.threat_detection_model.name
}

output "model_arn" {
  description = "ARN of the SageMaker model"
  value       = aws_sagemaker_model.threat_detection_model.arn
}

output "endpoint_config_name" {
  description = "Name of the SageMaker endpoint configuration"
  value       = aws_sagemaker_endpoint_configuration.threat_detection_config.name
}

output "endpoint_config_arn" {
  description = "ARN of the SageMaker endpoint configuration"
  value       = aws_sagemaker_endpoint_configuration.threat_detection_config.arn
}

output "endpoint_name" {
  description = "Name of the SageMaker endpoint"
  value       = aws_sagemaker_endpoint.threat_detection_endpoint.name
}

output "endpoint_arn" {
  description = "ARN of the SageMaker endpoint"
  value       = aws_sagemaker_endpoint.threat_detection_endpoint.arn
}

output "endpoint_url" {
  description = "URL for invoking the SageMaker endpoint"
  value       = "https://runtime.sagemaker.${var.aws_region}.amazonaws.com/endpoints/${aws_sagemaker_endpoint.threat_detection_endpoint.name}/invocations"
}

output "auto_scaling_target_arn" {
  description = "ARN of the auto scaling target"
  value       = var.enable_auto_scaling ? aws_appautoscaling_target.sagemaker_target[0].arn : null
}

output "auto_scaling_policy_arn" {
  description = "ARN of the auto scaling policy"
  value       = var.enable_auto_scaling ? aws_appautoscaling_policy.sagemaker_scaling_policy[0].arn : null
}

output "cloudwatch_alarms" {
  description = "Map of CloudWatch alarm names and ARNs"
  value = var.create_cloudwatch_alarms ? {
    invocation_errors = {
      name = aws_cloudwatch_metric_alarm.endpoint_invocation_errors[0].alarm_name
      arn  = aws_cloudwatch_metric_alarm.endpoint_invocation_errors[0].arn
    }
    model_latency = {
      name = aws_cloudwatch_metric_alarm.endpoint_model_latency[0].alarm_name
      arn  = aws_cloudwatch_metric_alarm.endpoint_model_latency[0].arn
    }
    low_invocations = {
      name = aws_cloudwatch_metric_alarm.endpoint_invocations[0].alarm_name
      arn  = aws_cloudwatch_metric_alarm.endpoint_invocations[0].arn
    }
  } : {}
}

output "endpoint_configuration" {
  description = "Complete endpoint configuration details"
  value = {
    endpoint_name        = aws_sagemaker_endpoint.threat_detection_endpoint.name
    endpoint_arn         = aws_sagemaker_endpoint.threat_detection_endpoint.arn
    endpoint_config_name = aws_sagemaker_endpoint_configuration.threat_detection_config.name
    model_name          = aws_sagemaker_model.threat_detection_model.name
    instance_type       = var.endpoint_instance_type
    instance_count      = var.endpoint_instance_count
    auto_scaling_enabled = var.enable_auto_scaling
  }
}