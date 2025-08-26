resource "aws_codebuild_project" "ml_build_project" {
  name         = var.project_name
  description  = "Build project for ${var.project_name} ML pipeline in ${var.environment} environment"
  service_role = var.service_role_arn
  tags         = var.tags

  # Artifacts
  artifacts {
    type     = var.artifacts_type
    location = var.artifacts_location
  }

  # Environment
  environment {
    compute_type                = var.compute_type
    image                       = var.build_image
    type                        = var.environment_type
    image_pull_credentials_type = var.image_pull_credentials_type
    privileged_mode             = var.privileged_mode

    # PLAINTEXT variables
    dynamic "environment_variable" {
      for_each = var.environment_variables
      content {
        name  = each.key
        value = each.value
        type  = "PLAINTEXT"
      }
    }

    # Parameter Store variables
    dynamic "environment_variable" {
      for_each = var.parameter_store_variables
      content {
        name  = each.key
        value = each.value
        type  = "PARAMETER_STORE"
      }
    }

    # Secrets Manager variables
    dynamic "environment_variable" {
      for_each = var.secrets_manager_variables
      content {
        name  = each.key
        value = each.value
        type  = "SECRETS_MANAGER"
      }
    }
  }

  # Source
  source {
    type            = var.source_type
    location        = var.source_location
    buildspec       = var.buildspec_file
    git_clone_depth = var.git_clone_depth

    dynamic "auth" {
      for_each = var.source_auth != null ? [var.source_auth] : []
      content {
        type     = each.value.type
        resource = each.value.resource
      }
    }
  }

  # VPC config
  dynamic "vpc_config" {
    for_each = var.vpc_config != null ? [var.vpc_config] : []
    content {
      vpc_id             = each.value.vpc_id
      subnets            = each.value.subnets
      security_group_ids = each.value.security_group_ids
    }
  }

  # Cache
  dynamic "cache" {
    for_each = var.cache_config != null ? [var.cache_config] : []
    content {
      type     = each.value.type
      location = each.value.location
      modes    = each.value.modes
    }
  }

  # Logs
  logs_config {
    dynamic "cloudwatch_logs" {
      for_each = var.cloudwatch_logs_config != null ? [var.cloudwatch_logs_config] : []
      content {
        status      = each.value.status
        group_name  = each.value.group_name
        stream_name = each.value.stream_name
      }
    }

    dynamic "s3_logs" {
      for_each = var.s3_logs_config != null ? [var.s3_logs_config] : []
      content {
        status              = each.value.status
        location            = each.value.location
        encryption_disabled = each.value.encryption_disabled
      }
    }
  }

  build_timeout           = var.build_timeout_minutes
  queued_timeout          = var.queued_timeout_minutes
  badge_enabled           = var.badge_enabled
  concurrent_build_limit  = var.concurrent_build_limit
}
