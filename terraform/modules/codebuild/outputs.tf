# CodeBuild Module Outputs

output "project_name" {
  description = "Name of the CodeBuild project"
  value       = aws_codebuild_project.ml_build_project.name
}

output "project_arn" {
  description = "ARN of the CodeBuild project"
  value       = aws_codebuild_project.ml_build_project.arn
}

output "project_id" {
  description = "ID of the CodeBuild project"
  value       = aws_codebuild_project.ml_build_project.id
}

output "badge_url" {
  description = "URL of the build badge"
  value       = aws_codebuild_project.ml_build_project.badge_url
}

output "webhook_url" {
  description = "Webhook URL for GitHub integration"
  value       = var.create_webhook ? aws_codebuild_webhook.build_webhook[0].url : null
}

output "webhook_secret" {
  description = "Webhook secret token"
  value       = var.create_webhook ? aws_codebuild_webhook.build_webhook[0].secret : null
  sensitive   = true
}

output "cloudwatch_log_group_name" {
  description = "Name of the CloudWatch log group"
  value       = var.create_cloudwatch_log_group ? aws_cloudwatch_log_group.codebuild_logs[0].name : null
}

output "cloudwatch_log_group_arn" {
  description = "ARN of the CloudWatch log group"
  value       = var.create_cloudwatch_log_group ? aws_cloudwatch_log_group.codebuild_logs[0].arn : null
}

output "build_project_details" {
  description = "Complete build project configuration"
  value = {
    name               = aws_codebuild_project.ml_build_project.name
    arn                = aws_codebuild_project.ml_build_project.arn
    compute_type       = var.compute_type
    build_image        = var.build_image
    environment_type   = var.environment_type
    build_timeout      = var.build_timeout_minutes
    queued_timeout     = var.queued_timeout_minutes
    privileged_mode    = var.privileged_mode
    badge_enabled      = var.badge_enabled
  }
}

output "alarm_arns" {
  description = "Map of CloudWatch alarm ARNs"
  value = {
    build_failures = var.create_failure_alarm ? aws_cloudwatch_metric_alarm.build_failures[0].arn : null
    build_duration = var.create_duration_alarm ? aws_cloudwatch_metric_alarm.build_duration[0].arn : null
  }
}