terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Environment = "prod"
      Project     = var.project_name
      ManagedBy   = "Terraform"
      Owner       = var.owner
      CostCenter  = "Production"
      Compliance  = "SOC2"
      DataClass   = "Confidential"
    }
  }
}

# Local values
locals {
  common_tags = {
    Environment    = "prod"
    Project        = var.project_name
    ManagedBy      = "Terraform"
    Owner          = var.owner
    CostCenter     = "Production"
    Compliance     = "SOC2"
    DataClass      = "Confidential"
    Backup         = "true"
    BusinessUnit   = var.business_unit
    ContactEmail   = var.contact_email
  }
}

# KMS Key for encryption
resource "aws_kms_key" "ml_key" {
  count                   = var.kms_key_id == "" ? 1 : 0
  description             = "KMS key for ${var.project_name} production encryption"
  deletion_window_in_days = 7
  tags                   = local.common_tags
}

resource "aws_kms_alias" "ml_key_alias" {
  count         = var.kms_key_id == "" ? 1 : 0
  name          = "alias/${var.project_name}-prod-key"
  target_key_id = aws_kms_key.ml_key[0].key_id
}

# SNS Topic for alerts
resource "aws_sns_topic" "ml_alerts" {
  count = var.sns_topic_arn == "" ? 1 : 0
  name  = "${var.project_name}-prod-alerts"
  tags  = local.common_tags
}

resource "aws_sns_topic_subscription" "email_alerts" {
  count     = var.sns_topic_arn == "" && var.alert_email != "" ? 1 : 0
  topic_arn = aws_sns_topic.ml_alerts[0].arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# S3 Module
module "s3" {
  source = "../../modules/s3"

  project_name = var.project_name
  environment  = "prod"
  tags         = local.common_tags

  # Production-specific S3 settings
  enable_versioning       = true
  enable_encryption      = true
  kms_key_id             = var.kms_key_id != "" ? var.kms_key_id : aws_kms_key.ml_key[0].arn
  enable_lifecycle       = true
  lifecycle_transition_days = var.lifecycle_transition_days
  data_retention_days    = var.data_retention_days
  model_retention_days   = var.model_retention_days
  create_pipeline_bucket = var.enable_cicd
  training_data_prefix   = var.training_data_prefix
  model_artifacts_prefix = var.model_artifacts_prefix
  sns_topic_arn         = var.sns_topic_arn != "" ? var.sns_topic_arn : (length(aws_sns_topic.ml_alerts) > 0 ? aws_sns_topic.ml_alerts[0].arn : null)
}

# IAM Module
module "iam" {
  source = "../../modules/iam"

  project_name = var.project_name
  environment  = "prod"
  tags         = local.common_tags

  # S3 bucket ARNs for permissions
  s3_bucket_arns = [
    module.s3.data_bucket_arn,
    module.s3.model_bucket_arn
  ]
  
  pipeline_s3_bucket_arn    = var.enable_cicd ? module.s3.pipeline_bucket_arn : null
  enable_vpc               = var.enable_vpc
  create_codebuild_role    = var.enable_cicd
  create_codepipeline_role = var.enable_cicd
}

# SageMaker Module
module "sagemaker" {
  source = "../../modules/sagemaker"

  project_name = var.project_name
  environment  = "prod"
  aws_region   = var.aws_region
  tags         = local.common_tags

  # IAM Configuration
  sagemaker_execution_role_arn = module.iam.sagemaker_execution_role_arn

  # Model Configuration
  model_data_url = "s3://${module.s3.model_bucket_name}/${var.model_artifacts_prefix}/model.tar.gz"

  # Production-specific SageMaker settings
  endpoint_instance_type  = var.endpoint_instance_type
  endpoint_instance_count = var.endpoint_instance_count
  
  # Auto Scaling
  enable_auto_scaling        = var.enable_auto_scaling
  auto_scaling_min_capacity  = var.auto_scaling_min_capacity
  auto_scaling_max_capacity  = var.auto_scaling_max_capacity
  auto_scaling_target_value  = var.auto_scaling_target_value

  # VPC Configuration
  vpc_config = var.enable_vpc ? {
    subnets            = var.subnet_ids
    security_group_ids = var.security_group_ids
  } : null

  # Data Capture for monitoring
  data_capture_config = {
    enable_capture              = true
    initial_sampling_percentage = 100
    destination_s3_uri         = "s3://${module.s3.model_bucket_name}/data-capture/"
    capture_options            = ["Input", "Output"]
  }

  # Encryption
  kms_key_id = var.kms_key_id != "" ? var.kms_key_id : aws_kms_key.ml_key[0].arn

  # Blue/Green Deployment
  blue_green_update_policy = {
    traffic_routing_configuration = {
      type                     = "AllAtOnce"
      wait_interval_in_seconds = 0
    }
    termination_wait_in_seconds          = 600
    maximum_execution_timeout_in_seconds = 14400
  }

  # CloudWatch Alarms
  create_cloudwatch_alarms = true
  alarm_actions = [
    var.sns_topic_arn != "" ? var.sns_topic_arn : aws_sns_topic.ml_alerts[0].arn
  ]
  invocation_error_threshold = var.invocation_error_threshold
  model_latency_threshold    = var.model_latency_threshold
  low_invocations_threshold  = var.low_invocations_threshold
}

# CodeBuild Module (conditional)
module "codebuild" {
  count  = var.enable_cicd ? 1 : 0
  source = "../../modules/codebuild"

  project_name = var.project_name
  environment  = "prod"
  tags         = local.common_tags
  source_type = var.source_type
  source_location = var.source_location
  service_role_arn = module.iam.codebuild_service_role_arn

  # Production-specific CodeBuild settings
  compute_type = var.codebuild_compute_type
  build_image  = "aws/codebuild/amazonlinux2-x86_64-standard:4.0"

  # Environment variables
  environment_variables = {
    AWS_DEFAULT_REGION     = var.aws_region
    ENVIRONMENT           = "prod"
    DATA_BUCKET           = module.s3.data_bucket_name
    MODEL_BUCKET          = module.s3.model_bucket_name
    SAGEMAKER_ROLE_ARN    = module.iam.sagemaker_execution_role_arn
    ENDPOINT_NAME         = module.sagemaker.endpoint_name
    PROJECT_NAME          = var.project_name
    KMS_KEY_ID           = var.kms_key_id != "" ? var.kms_key_id : aws_kms_key.ml_key[0].arn
  }

  # Production build configuration
  buildspec_file         = "buildspec-prod.yml"
  build_timeout_minutes  = var.build_timeout_minutes
  queued_timeout_minutes = 960  # Longer queue timeout for prod

  # VPC Configuration
  vpc_config = var.enable_vpc ? {
    vpc_id             = var.vpc_id
    subnets            = var.subnet_ids
    security_group_ids = var.security_group_ids
  } : null

  # Logs with encryption
  create_cloudwatch_log_group = true
  log_retention_days         = var.log_retention_days
  logs_kms_key_id           = var.kms_key_id != "" ? var.kms_key_id : aws_kms_key.ml_key[0].arn

  # Cache configuration
  cache_config = {
    type     = "S3"
    location = "${module.s3.pipeline_bucket_name}/cache"
  }

  # Alarms
  create_failure_alarm     = true
  create_duration_alarm    = true
  alarm_actions = [
    var.sns_topic_arn != "" ? var.sns_topic_arn : aws_sns_topic.ml_alerts[0].arn
  ]
  duration_threshold_minutes = var.build_duration_threshold_minutes
}

# CodePipeline Module (conditional)
module "codepipeline" {
  count  = var.enable_cicd && var.github_repo != "" && var.github_token != "" ? 1 : 0
  source = "../../modules/codepipeline"

  project_name = var.project_name
  environment  = "prod"
  tags         = local.common_tags

  service_role_arn      = module.iam.codepipeline_service_role_arn
  artifact_store_bucket = module.s3.pipeline_bucket_name
  artifact_store_kms_key_id = var.kms_key_id != "" ? var.kms_key_id : aws_kms_key.ml_key[0].arn

  # Source actions
  source_actions = [
    {
      name             = "Source"
      owner            = "ThirdParty"
      provider         = "GitHub"
      version          = "1"
      output_artifacts = ["source_output"]
      configuration = {
        Owner      = split("/", var.github_repo)[0]
        Repo       = split("/", var.github_repo)[1]
        Branch     = var.github_branch
        OAuthToken = var.github_token
      }
    }
  ]

  # Build actions
  build_actions = [
    {
      name             = "Build"
      owner            = "AWS"
      provider         = "CodeBuild"
      version          = "1"
      input_artifacts  = ["source_output"]
      output_artifacts = ["build_output"]
      configuration = {
        ProjectName = module.codebuild[0].project_name
      }
    }
  ]

  # Deploy actions with manual approval for production
  deploy_actions = [
    {
      name      = "ApprovalRequired"
      category  = "Approval"
      owner     = "AWS"
      provider  = "Manual"
      version   = "1"
      run_order = 1
      configuration = {
        CustomData = "Please review the model performance metrics and approve for production deployment"
      }
    },
    {
      name      = "DeployToProduction"
      category  = "Invoke"
      owner     = "AWS"
      provider  = "Lambda"
      version   = "1"
      run_order = 2
      input_artifacts = ["build_output"]
      configuration = {
        FunctionName = "deploy-sagemaker-model-${var.project_name}-prod"
      }
    }
  ]

  # Production-specific pipeline settings
  create_failure_alarm     = true
  create_duration_alarm    = true
  alarm_actions = [
    var.sns_topic_arn != "" ? var.sns_topic_arn : aws_sns_topic.ml_alerts[0].arn
  ]
  duration_threshold_minutes = var.pipeline_duration_threshold_minutes

  # Events
  create_pipeline_events = true
  event_target_arn = var.sns_topic_arn != "" ? var.sns_topic_arn : aws_sns_topic.ml_alerts[0].arn
  event_input_transformer = {
    input_paths = {
      pipeline = "$.detail.pipeline"
      state    = "$.detail.state"
    }
    input_template = "Pipeline <pipeline> has changed state to <state>"
  }
}

# CloudWatch Dashboard for production monitoring
resource "aws_cloudwatch_dashboard" "ml_dashboard" {
  dashboard_name = "${var.project_name}-prod-dashboard"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        properties = {
          metrics = [
            ["AWS/SageMaker", "Invocations", "EndpointName", module.sagemaker.endpoint_name],
            ["AWS/SageMaker", "ModelLatency", "EndpointName", module.sagemaker.endpoint_name],
            ["AWS/SageMaker", "Invocation4XXErrors", "EndpointName", module.sagemaker.endpoint_name],
            ["AWS/SageMaker", "Invocation5XXErrors", "EndpointName", module.sagemaker.endpoint_name]
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "SageMaker Endpoint Metrics"
          period  = 300
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 12
        height = 6
        properties = {
          metrics = var.enable_cicd ? [
            ["AWS/CodeBuild", "Duration", "ProjectName", module.codebuild[0].project_name],
            ["AWS/CodeBuild", "SucceededBuilds", "ProjectName", module.codebuild[0].project_name],
            ["AWS/CodeBuild", "FailedBuilds", "ProjectName", module.codebuild[0].project_name]
          ] : []
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "CodeBuild Metrics"
          period  = 300
        }
      }
    ]
  })
}