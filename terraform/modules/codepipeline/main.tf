# CodePipeline Module for Cybersecurity Threat Detection System

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Local values
locals {
  pipeline_name = "${var.project_name}-${var.environment}-pipeline"
}

# CodePipeline
resource "aws_codepipeline" "ml_pipeline" {
  name     = local.pipeline_name
  role_arn = var.service_role_arn
  tags     = var.tags

  artifact_store {
    location = var.artifact_store_bucket
    type     = "S3"

    # Encryption configuration
    dynamic "encryption_key" {
      for_each = var.artifact_store_kms_key_id != null ? [var.artifact_store_kms_key_id] : []
      content {
        id   = encryption_key.value
        type = "KMS"
      }
    }
  }

  # Source Stage
  stage {
    name = "Source"

    dynamic "action" {
      for_each = var.source_actions
      content {
        name             = action.value.name
        category         = "Source"
        owner            = action.value.owner
        provider         = action.value.provider
        version          = action.value.version
        output_artifacts = action.value.output_artifacts
        run_order        = action.value.run_order

        configuration = action.value.configuration
        
        # Input artifacts (for some source types)
        input_artifacts = action.value.input_artifacts
        
        # Namespace for action variables
        namespace = action.value.namespace

        # Region override
        region = action.value.region
      }
    }
  }

  # Build Stage
  dynamic "stage" {
    for_each = var.build_actions != null ? [var.build_actions] : []
    content {
      name = "Build"

      dynamic "action" {
        for_each = stage.value
        content {
          name             = action.value.name
          category         = "Build"
          owner            = action.value.owner
          provider         = action.value.provider
          version          = action.value.version
          input_artifacts  = action.value.input_artifacts
          output_artifacts = action.value.output_artifacts
          run_order        = action.value.run_order

          configuration = action.value.configuration
          
          namespace = action.value.namespace
          region    = action.value.region
        }
      }
    }
  }

  # Test Stage
  dynamic "stage" {
    for_each = var.test_actions != null ? [var.test_actions] : []
    content {
      name = "Test"

      dynamic "action" {
        for_each = stage.value
        content {
          name             = action.value.name
          category         = action.value.category
          owner            = action.value.owner
          provider         = action.value.provider
          version          = action.value.version
          input_artifacts  = action.value.input_artifacts
          output_artifacts = action.value.output_artifacts
          run_order        = action.value.run_order

          configuration = action.value.configuration
          
          namespace = action.value.namespace
          region    = action.value.region
        }
      }
    }
  }

  # Deploy Stage
  dynamic "stage" {
    for_each = var.deploy_actions != null ? [var.deploy_actions] : []
    content {
      name = "Deploy"

      dynamic "action" {
        for_each = stage.value
        content {
          name             = action.value.name
          category         = action.value.category
          owner            = action.value.owner
          provider         = action.value.provider
          version          = action.value.version
          input_artifacts  = action.value.input_artifacts
          output_artifacts = action.value.output_artifacts
          run_order        = action.value.run_order

          configuration = action.value.configuration
          
          namespace = action.value.namespace
          region    = action.value.region
        }
      }
    }
  }

  # Custom Stages
  dynamic "stage" {
    for_each = var.custom_stages
    content {
      name = stage.value.name

      dynamic "action" {
        for_each = stage.value.actions
        content {
          name             = action.value.name
          category         = action.value.category
          owner            = action.value.owner
          provider         = action.value.provider
          version          = action.value.version
          input_artifacts  = action.value.input_artifacts
          output_artifacts = action.value.output_artifacts
          run_order        = action.value.run_order

          configuration = action.value.configuration
          
          namespace = action.value.namespace
          region    = action.value.region
        }
      }
    }
  }
}

# CloudWatch Event Rule for pipeline state changes
resource "aws_cloudwatch_event_rule" "pipeline_state_change" {
  count       = var.create_pipeline_events ? 1 : 0
  name        = "${local.pipeline_name}-state-change"
  description = "Capture pipeline state changes"
  tags        = var.tags

  event_pattern = jsonencode({
    source      = ["aws.codepipeline"]
    detail-type = ["CodePipeline Pipeline Execution State Change"]
    detail = {
      pipeline = [aws_codepipeline.ml_pipeline.name]
    }
  })
}

# CloudWatch Event Target
resource "aws_cloudwatch_event_target" "pipeline_event_target" {
  count     = var.create_pipeline_events && var.event_target_arn != null ? 1 : 0
  rule      = aws_cloudwatch_event_rule.pipeline_state_change[0].name
  target_id = "SendToTarget"
  arn       = var.event_target_arn

  # Input transformer for custom message format
  dynamic "input_transformer" {
    for_each = var.event_input_transformer != null ? [var.event_input_transformer] : []
    content {
      input_paths    = input_transformer.value.input_paths
      input_template = input_transformer.value.input_template
    }
  }
}

# CloudWatch Alarms for pipeline failures
resource "aws_cloudwatch_metric_alarm" "pipeline_failed" {
  count               = var.create_failure_alarm ? 1 : 0
  alarm_name          = "${local.pipeline_name}-failed"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = var.alarm_evaluation_periods
  metric_name         = "PipelineExecutionFailure"
  namespace           = "AWS/CodePipeline"
  period              = var.alarm_period
  statistic           = "Sum"
  threshold           = var.failure_threshold
  alarm_description   = "This metric monitors CodePipeline execution failures"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.alarm_actions
  treat_missing_data  = "notBreaching"
  tags                = var.tags

  dimensions = {
    PipelineName = aws_codepipeline.ml_pipeline.name
  }
}

resource "aws_cloudwatch_metric_alarm" "pipeline_duration" {
  count               = var.create_duration_alarm ? 1 : 0
  alarm_name          = "${local.pipeline_name}-duration"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = var.alarm_evaluation_periods
  metric_name         = "PipelineExecutionDuration"
  namespace           = "AWS/CodePipeline"
  period              = var.alarm_period
  statistic           = "Average"
  threshold           = var.duration_threshold_minutes * 60 # Convert to seconds
  alarm_description   = "This metric monitors CodePipeline execution duration"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.alarm_actions
  treat_missing_data  = "notBreaching"
  tags                = var.tags

  dimensions = {
    PipelineName = aws_codepipeline.ml_pipeline.name
  }
}