# CodeBuild Module for Cybersecurity Threat Detection System

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
  project_name = "${var.project_name}-${var.environment}-build"
}

# CodeBuild Project
resource "aws_codebuild_project" "ml_build_project" {
  name          = local.project_name
  description   = "Build project for ${var.project_name} ML pipeline in ${var.environment} environment"
  service_role  = var.service_role_arn
  tags          = var.tags

  artifacts {
    type = var.artifacts_type
    
    dynamic "location" {
      for_each = var.artifacts_location != null ? [var.artifacts_location] : []
      content {
        location = artifacts_location.value
      }
    }
  }

  environment {
    compute_type                = var.compute_type
    image                      = var.build_image
    type                       = var.environment_type
    image_pull_credentials_type = var.image_pull_credentials_type
    privileged_mode            = var.privileged_mode

    # Environment variables
    dynamic "environment_variable" {
      for_each = var.environment_variables
      content {
        name  = environment_variable.key
        value = environment_variable.value
        type  = "PLAINTEXT"
      }
    }

    # Parameter Store environment variables
    dynamic "environment_variable" {
      for_each = var.parameter_store_variables
      content {
        name  = environment_variable.key
        value = environment_variable.value
        type  = "PARAMETER_STORE"
      }
    }

    # Secrets Manager environment variables
    dynamic "environment_variable" {
      for_each = var.secrets_manager_variables
      content {
        name  = environment_variable.key
        value = environment_variable.value
        type  = "SECRETS_MANAGER"
      }
    }
  }

  source {
    type            = var.source_type
    location        = var.source_location
    buildspec       = var.buildspec_file
    git_clone_depth = var.git_clone_depth

    # Git submodules config
    dynamic "git_submodules_config" {
      for_each = var.fetch_git_submodules ? [1] : []
      content {
        fetch_submodules = true
      }
    }

    # Auth configuration for private repositories
    dynamic "auth" {
      for_each = var.source_auth != null ? [var.source_auth] : []
      content {
        type     = auth.value.type
        resource = auth.value.resource
      }
    }
  }

  # Secondary sources
  dynamic "secondary_sources" {
    for_each = var.secondary_sources
    content {
      type              = secondary_sources.value.type
      location          = secondary_sources.value.location
      source_identifier = secondary_sources.value.source_identifier
      buildspec         = secondary_sources.value.buildspec
      git_clone_depth   = secondary_sources.value.git_clone_depth
      
      dynamic "git_submodules_config" {
        for_each = secondary_sources.value.fetch_git_submodules ? [1] : []
        content {
          fetch_submodules = true
        }
      }
    }
  }

  # VPC configuration
  dynamic "vpc_config" {
    for_each = var.vpc_config != null ? [var.vpc_config] : []
    content {
      vpc_id             = vpc_config.value.vpc_id
      subnets            = vpc_config.value.subnets
      security_group_ids = vpc_config.value.security_group_ids
    }
  }

  # Cache configuration
  dynamic "cache" {
    for_each = var.cache_config != null ? [var.cache_config] : []
    content {
      type     = cache.value.type
      location = cache.value.location
      modes    = cache.value.modes
    }
  }

  # Logs configuration
  logs_config {
    # CloudWatch Logs
    dynamic "cloudwatch_logs" {
      for_each = var.cloudwatch_logs_config != null ? [var.cloudwatch_logs_config] : []
      content {
        status      = cloudwatch_logs.value.status
        group_name  = cloudwatch_logs.value.group_name
        stream_name = cloudwatch_logs.value.stream_name
      }
    }

    # S3 Logs
    dynamic "s3_logs" {
      for_each = var.s3_logs_config != null ? [var.s3_logs_config] : []
      content {
        status              = s3_logs.value.status
        location            = s3_logs.value.location
        encryption_disabled = s3_logs.value.encryption_disabled
      }
    }
  }

  # Build timeout
  build_timeout = var.build_timeout_minutes

  # Queued timeout
  queued_timeout = var.queued_timeout_minutes

  # Badge
  badge_enabled = var.badge_enabled

  # Concurrent builds
  concurrent_build_limit = var.concurrent_build_limit
}

# CloudWatch Log Group for CodeBuild
resource "aws_cloudwatch_log_group" "codebuild_logs" {
  count             = var.create_cloudwatch_log_group ? 1 : 0
  name              = "/aws/codebuild/${local.project_name}"
  retention_in_days = var.log_retention_days
  tags              = var.tags

  kms_key_id = var.logs_kms_key_id
}

# CodeBuild Webhook (for GitHub integration)
resource "aws_codebuild_webhook" "build_webhook" {
  count        = var.create_webhook ? 1 : 0
  project_name = aws_codebuild_project.ml_build_project.name
  build_type   = var.webhook_build_type

  dynamic "filter_group" {
    for_each = var.webhook_filters
    content {
      dynamic "filter" {
        for_each = filter_group.value
        content {
          type    = filter.value.type
          pattern = filter.value.pattern
        }
      }
    }
  }
}

# CloudWatch Alarms for build failures
resource "aws_cloudwatch_metric_alarm" "build_failures" {
  count               = var.create_failure_alarm ? 1 : 0
  alarm_name          = "${local.project_name}-build-failures"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = var.alarm_evaluation_periods
  metric_name         = "FailedBuilds"
  namespace           = "AWS/CodeBuild"
  period              = var.alarm_period
  statistic           = "Sum"
  threshold           = var.failure_threshold
  alarm_description   = "This metric monitors CodeBuild project failures"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.alarm_actions
  treat_missing_data  = "notBreaching"
  tags                = var.tags

  dimensions = {
    ProjectName = aws_codebuild_project.ml_build_project.name
  }
}

resource "aws_cloudwatch_metric_alarm" "build_duration" {
  count               = var.create_duration_alarm ? 1 : 0
  alarm_name          = "${local.project_name}-build-duration"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = var.alarm_evaluation_periods
  metric_name         = "Duration"
  namespace           = "AWS/CodeBuild"
  period              = var.alarm_period
  statistic           = "Average"
  threshold           = var.duration_threshold_minutes * 60 # Convert to seconds
  alarm_description   = "This metric monitors CodeBuild project duration"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.alarm_actions
  treat_missing_data  = "notBreaching"
  tags                = var.tags

  dimensions = {
    ProjectName = aws_codebuild_project.ml_build_project.name
  }
}