# SageMaker Module for Cybersecurity Threat Detection System

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
  model_name = "${var.project_name}-${var.environment}-model"
  endpoint_config_name = "${var.project_name}-${var.environment}-endpoint-config-${formatdate("YYYYMMDD-hhmmss", timestamp())}"
  endpoint_name = "${var.project_name}-${var.environment}-endpoint"
}

# SageMaker Model
resource "aws_sagemaker_model" "threat_detection_model" {
  name               = local.model_name
  execution_role_arn = var.sagemaker_execution_role_arn
  tags               = var.tags

  primary_container {
    image          = var.training_image_uri
    model_data_url = var.model_data_url
    
    environment = merge(
      var.model_environment_variables,
      {
        SAGEMAKER_PROGRAM                = var.inference_script_name
        SAGEMAKER_SUBMIT_DIRECTORY       = "/opt/ml/code"
        SAGEMAKER_CONTAINER_LOG_LEVEL    = var.log_level
        SAGEMAKER_REGION                 = var.aws_region
      }
    )
  }

  # VPC configuration if specified
  dynamic "vpc_config" {
    for_each = var.vpc_config != null ? [var.vpc_config] : []
    content {
      subnets            = vpc_config.value.subnets
      security_group_ids = vpc_config.value.security_group_ids
    }
  }
}

# SageMaker Endpoint Configuration
resource "aws_sagemaker_endpoint_configuration" "threat_detection_config" {
  name = local.endpoint_config_name
  tags = var.tags

  production_variants {
    variant_name           = "primary"
    model_name            = aws_sagemaker_model.threat_detection_model.name
    initial_instance_count = var.endpoint_instance_count
    instance_type         = var.endpoint_instance_type
    initial_variant_weight = 1
  }

  # Data capture configuration
  dynamic "data_capture_config" {
    for_each = var.data_capture_config != null ? [var.data_capture_config] : []
    content {
      enable_capture              = data_capture_config.value.enable_capture
      initial_sampling_percentage = data_capture_config.value.initial_sampling_percentage
      destination_s3_uri         = data_capture_config.value.destination_s3_uri
      
      dynamic "capture_options" {
        for_each = data_capture_config.value.capture_options
        content {
          capture_mode = capture_options.value
        }
      }

      dynamic "capture_content_type_header" {
        for_each = data_capture_config.value.capture_content_type_header != null ? [data_capture_config.value.capture_content_type_header] : []
        content {
          csv_content_types  = capture_content_type_header.value.csv_content_types
          json_content_types = capture_content_type_header.value.json_content_types
        }
      }
    }
  }

  # Server-side encryption
  dynamic "kms_key_id" {
    for_each = var.kms_key_id != null ? [var.kms_key_id] : []
    content {
      kms_key_id = kms_key_id.value
    }
  }

  lifecycle {
    create_before_destroy = true
  }
}

# SageMaker Endpoint
resource "aws_sagemaker_endpoint" "threat_detection_endpoint" {
  name                 = local.endpoint_name
  endpoint_config_name = aws_sagemaker_endpoint_configuration.threat_detection_config.name
  tags                 = var.tags

  deployment_config {
    dynamic "blue_green_update_policy" {
      for_each = var.blue_green_update_policy != null ? [var.blue_green_update_policy] : []
      content {
        traffic_routing_configuration {
          type                     = blue_green_update_policy.value.traffic_routing_configuration.type
          wait_interval_in_seconds = blue_green_update_policy.value.traffic_routing_configuration.wait_interval_in_seconds
          
          dynamic "canary_size" {
            for_each = blue_green_update_policy.value.traffic_routing_configuration.canary_size != null ? [blue_green_update_policy.value.traffic_routing_configuration.canary_size] : []
            content {
              type  = canary_size.value.type
              value = canary_size.value.value
            }
          }
        }
        
        termination_wait_in_seconds     = blue_green_update_policy.value.termination_wait_in_seconds
        maximum_execution_timeout_in_seconds = blue_green_update_policy.value.maximum_execution_timeout_in_seconds
      }
    }

    auto_rollback_configuration {
      dynamic "alarms" {
        for_each = var.auto_rollback_alarms
        content {
          alarm_name = alarms.value
        }
      }
    }
  }
}

# Auto Scaling Target (if enabled)
resource "aws_appautoscaling_target" "sagemaker_target" {
  count              = var.enable_auto_scaling ? 1 : 0
  max_capacity       = var.auto_scaling_max_capacity
  min_capacity       = var.auto_scaling_min_capacity
  resource_id        = "endpoint/${aws_sagemaker_endpoint.threat_detection_endpoint.name}/variant/primary"
  scalable_dimension = "sagemaker:variant:DesiredInstanceCount"
  service_namespace  = "sagemaker"
  tags               = var.tags
}

# Auto Scaling Policy
resource "aws_appautoscaling_policy" "sagemaker_scaling_policy" {
  count              = var.enable_auto_scaling ? 1 : 0
  name               = "${local.endpoint_name}-scaling-policy"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.sagemaker_target[0].resource_id
  scalable_dimension = aws_appautoscaling_target.sagemaker_target[0].scalable_dimension
  service_namespace  = aws_appautoscaling_target.sagemaker_target[0].service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = var.auto_scaling_metric_type
    }
    target_value       = var.auto_scaling_target_value
    scale_in_cooldown  = var.auto_scaling_scale_in_cooldown
    scale_out_cooldown = var.auto_scaling_scale_out_cooldown
  }
}

# CloudWatch Alarms for monitoring
resource "aws_cloudwatch_metric_alarm" "endpoint_invocation_errors" {
  count               = var.create_cloudwatch_alarms ? 1 : 0
  alarm_name          = "${local.endpoint_name}-invocation-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = var.alarm_evaluation_periods
  metric_name         = "Invocation4XXErrors"
  namespace           = "AWS/SageMaker"
  period              = var.alarm_period
  statistic           = "Sum"
  threshold           = var.invocation_error_threshold
  alarm_description   = "This metric monitors SageMaker endpoint 4XX errors"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.alarm_actions
  treat_missing_data  = "notBreaching"
  tags                = var.tags

  dimensions = {
    EndpointName = aws_sagemaker_endpoint.threat_detection_endpoint.name
    VariantName  = "primary"
  }
}

resource "aws_cloudwatch_metric_alarm" "endpoint_model_latency" {
  count               = var.create_cloudwatch_alarms ? 1 : 0
  alarm_name          = "${local.endpoint_name}-model-latency"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = var.alarm_evaluation_periods
  metric_name         = "ModelLatency"
  namespace           = "AWS/SageMaker"
  period              = var.alarm_period
  statistic           = "Average"
  threshold           = var.model_latency_threshold
  alarm_description   = "This metric monitors SageMaker endpoint model latency"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.alarm_actions
  treat_missing_data  = "notBreaching"
  tags                = var.tags

  dimensions = {
    EndpointName = aws_sagemaker_endpoint.threat_detection_endpoint.name
    VariantName  = "primary"
  }
}

resource "aws_cloudwatch_metric_alarm" "endpoint_invocations" {
  count               = var.create_cloudwatch_alarms ? 1 : 0
  alarm_name          = "${local.endpoint_name}-low-invocations"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = var.alarm_evaluation_periods
  metric_name         = "Invocations"
  namespace           = "AWS/SageMaker"
  period              = var.alarm_period
  statistic           = "Sum"
  threshold           = var.low_invocations_threshold
  alarm_description   = "This metric monitors SageMaker endpoint for low activity"
  alarm_actions       = var.alarm_actions
  ok_actions          = var.alarm_actions
  treat_missing_data  = "notBreaching"
  tags                = var.tags

  dimensions = {
    EndpointName = aws_sagemaker_endpoint.threat_detection_endpoint.name
    VariantName  = "primary"
  }
}