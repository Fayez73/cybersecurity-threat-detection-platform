# CodePipeline Module Outputs

output "pipeline_name" {
  description = "Name of the CodePipeline"
  value       = aws_codepipeline.ml_pipeline.name
}

output "pipeline_arn" {
  description = "ARN of the CodePipeline"
  value       = aws_codepipeline.ml_pipeline.arn
}

output "pipeline_id" {
  description = "ID of the CodePipeline"
  value       = aws_codepipeline.ml_pipeline.id
}

output "pipeline_url" {
  description = "URL of the CodePipeline in AWS Console"
  value       = "https://console.aws.amazon.com/codesuite/codepipeline/pipelines/${aws_codepipeline.ml_pipeline.name}/view"
}

output "event_rule_arn" {
  description = "ARN of the CloudWatch Event Rule"
  value       = var.create_pipeline_events ? aws_cloudwatch_event_rule.pipeline_state_change[0].arn : null
}

output "event_rule_name" {
  description = "Name of the CloudWatch Event Rule"
  value       = var.create_pipeline_events ? aws_cloudwatch_event_rule.pipeline_state_change[0].name : null
}

output "pipeline_configuration" {
  description = "Complete pipeline configuration details"
  value = {
    name                = aws_codepipeline.ml_pipeline.name
    arn                 = aws_codepipeline.ml_pipeline.arn
    artifact_store      = var.artifact_store_bucket
    source_actions      = length(var.source_actions)
    build_actions       = var.build_actions != null ? length(var.build_actions) : 0
    test_actions        = var.test_actions != null ? length(var.test_actions) : 0
    deploy_actions      = var.deploy_actions != null ? length(var.deploy_actions) : 0
    custom_stages       = length(var.custom_stages)
    events_enabled      = var.create_pipeline_events
  }
}

output "alarm_arns" {
  description = "Map of CloudWatch alarm ARNs"
  value = {
    pipeline_failures = var.create_failure_alarm ? aws_cloudwatch_metric_alarm.pipeline_failed[0].arn : null
    pipeline_duration = var.create_duration_alarm ? aws_cloudwatch_metric_alarm.pipeline_duration[0].arn : null
  }
}