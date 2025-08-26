# IAM Module for Cybersecurity Threat Detection System

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# SageMaker Execution Role
resource "aws_iam_role" "sagemaker_execution_role" {
  name               = "${var.project_name}-${var.environment}-sagemaker-execution-role"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume_role.json
  tags               = var.tags
}

data "aws_iam_policy_document" "sagemaker_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    effect  = "Allow"

    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

# SageMaker Execution Policy
resource "aws_iam_policy" "sagemaker_execution_policy" {
  name        = "${var.project_name}-${var.environment}-sagemaker-execution-policy"
  description = "IAM policy for SageMaker execution role"
  policy      = data.aws_iam_policy_document.sagemaker_execution_policy.json
  tags        = var.tags
}

data "aws_iam_policy_document" "sagemaker_execution_policy" {
  # S3 permissions for data and model buckets
  statement {
    sid    = "S3Access"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket",
      "s3:GetBucketLocation"
    ]
    resources = concat(
      var.s3_bucket_arns,
      [for arn in var.s3_bucket_arns : "${arn}/*"]
    )
  }

  # CloudWatch Logs permissions
  statement {
    sid    = "CloudWatchLogs"
    effect = "Allow"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogStreams",
      "logs:DescribeLogGroups"
    ]
    resources = [
      "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/aws/sagemaker/*"
    ]
  }

  # ECR permissions for container images
  statement {
    sid    = "ECRAccess"
    effect = "Allow"
    actions = [
      "ecr:GetAuthorizationToken",
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage"
    ]
    resources = ["*"]
  }

  # CloudWatch metrics permissions
  statement {
    sid    = "CloudWatchMetrics"
    effect = "Allow"
    actions = [
      "cloudwatch:PutMetricData"
    ]
    resources = ["*"]
    
    condition {
      test     = "StringEquals"
      variable = "cloudwatch:namespace"
      values   = ["AWS/SageMaker"]
    }
  }

  # VPC permissions (if VPC is enabled)
  dynamic "statement" {
    for_each = var.enable_vpc ? [1] : []
    content {
      sid    = "VPCAccess"
      effect = "Allow"
      actions = [
        "ec2:CreateNetworkInterface",
        "ec2:CreateNetworkInterfacePermission",
        "ec2:DeleteNetworkInterface",
        "ec2:DeleteNetworkInterfacePermission",
        "ec2:DescribeNetworkInterfaces",
        "ec2:DescribeVpcs",
        "ec2:DescribeDhcpOptions",
        "ec2:DescribeSubnets",
        "ec2:DescribeSecurityGroups"
      ]
      resources = ["*"]
    }
  }
}

resource "aws_iam_role_policy_attachment" "sagemaker_execution_policy" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = aws_iam_policy.sagemaker_execution_policy.arn
}

# CodeBuild Service Role (conditional)
resource "aws_iam_role" "codebuild_service_role" {
  count              = var.create_codebuild_role ? 1 : 0
  name               = "${var.project_name}-${var.environment}-codebuild-service-role"
  assume_role_policy = data.aws_iam_policy_document.codebuild_assume_role[0].json
  tags               = var.tags
}

data "aws_iam_policy_document" "codebuild_assume_role" {
  count = var.create_codebuild_role ? 1 : 0
  
  statement {
    actions = ["sts:AssumeRole"]
    effect  = "Allow"

    principals {
      type        = "Service"
      identifiers = ["codebuild.amazonaws.com"]
    }
  }
}

# CodeBuild Service Policy
resource "aws_iam_policy" "codebuild_service_policy" {
  count       = var.create_codebuild_role ? 1 : 0
  name        = "${var.project_name}-${var.environment}-codebuild-service-policy"
  description = "IAM policy for CodeBuild service role"
  policy      = data.aws_iam_policy_document.codebuild_service_policy[0].json
  tags        = var.tags
}

data "aws_iam_policy_document" "codebuild_service_policy" {
  count = var.create_codebuild_role ? 1 : 0

  # CloudWatch Logs permissions
  statement {
    sid    = "CloudWatchLogs"
    effect = "Allow"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents"
    ]
    resources = [
      "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/aws/codebuild/${var.project_name}-${var.environment}-*"
    ]
  }

  # S3 permissions
  statement {
    sid    = "S3Access"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:GetObjectVersion",
      "s3:PutObject",
      "s3:ListBucket"
    ]
    resources = concat(
      var.s3_bucket_arns,
      [for arn in var.s3_bucket_arns : "${arn}/*"]
    )
  }

  # SageMaker permissions
  statement {
    sid    = "SageMakerAccess"
    effect = "Allow"
    actions = [
      "sagemaker:CreateTrainingJob",
      "sagemaker:DescribeTrainingJob",
      "sagemaker:CreateModel",
      "sagemaker:CreateEndpointConfig",
      "sagemaker:CreateEndpoint",
      "sagemaker:UpdateEndpoint",
      "sagemaker:DescribeEndpoint",
      "sagemaker:DescribeEndpointConfig",
      "sagemaker:DescribeModel",
      "sagemaker:DeleteModel",
      "sagemaker:DeleteEndpointConfig",
      "sagemaker:DeleteEndpoint",
      "sagemaker:ListTags",
      "sagemaker:AddTags"
    ]
    resources = [
      "arn:aws:sagemaker:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:*/${var.project_name}-${var.environment}-*"
    ]
  }

  # IAM PassRole permission
  statement {
    sid    = "PassRole"
    effect = "Allow"
    actions = [
      "iam:PassRole"
    ]
    resources = [
      aws_iam_role.sagemaker_execution_role.arn
    ]
  }
}

resource "aws_iam_role_policy_attachment" "codebuild_service_policy" {
  count      = var.create_codebuild_role ? 1 : 0
  role       = aws_iam_role.codebuild_service_role[0].name
  policy_arn = aws_iam_policy.codebuild_service_policy[0].arn
}

# CodePipeline Service Role (conditional)
resource "aws_iam_role" "codepipeline_service_role" {
  count              = var.create_codepipeline_role ? 1 : 0
  name               = "${var.project_name}-${var.environment}-codepipeline-service-role"
  assume_role_policy = data.aws_iam_policy_document.codepipeline_assume_role[0].json
  tags               = var.tags
}

data "aws_iam_policy_document" "codepipeline_assume_role" {
  count = var.create_codepipeline_role ? 1 : 0
  
  statement {
    actions = ["sts:AssumeRole"]
    effect  = "Allow"

    principals {
      type        = "Service"
      identifiers = ["codepipeline.amazonaws.com"]
    }
  }
}

# CodePipeline Service Policy
resource "aws_iam_policy" "codepipeline_service_policy" {
  count       = var.create_codepipeline_role ? 1 : 0
  name        = "${var.project_name}-${var.environment}-codepipeline-service-policy"
  description = "IAM policy for CodePipeline service role"
  policy      = data.aws_iam_policy_document.codepipeline_service_policy[0].json
  tags        = var.tags
}

data "aws_iam_policy_document" "codepipeline_service_policy" {
  count = var.create_codepipeline_role ? 1 : 0

  # S3 permissions for artifacts
  statement {
    sid    = "S3ArtifactAccess"
    effect = "Allow"
    actions = [
      "s3:GetBucketVersioning",
      "s3:GetObject",
      "s3:GetObjectVersion",
      "s3:PutObject",
      "s3:ListBucket"
    ]
    resources = var.pipeline_s3_bucket_arn != null ? [
      var.pipeline_s3_bucket_arn,
      "${var.pipeline_s3_bucket_arn}/*"
    ] : []
  }

  # CodeBuild permissions
  statement {
    sid    = "CodeBuildAccess"
    effect = "Allow"
    actions = [
      "codebuild:BatchGetBuilds",
      "codebuild:StartBuild"
    ]
    resources = [
      "arn:aws:codebuild:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:project/${var.project_name}-${var.environment}-*"
    ]
  }
}

resource "aws_iam_role_policy_attachment" "codepipeline_service_policy" {
  count      = var.create_codepipeline_role ? 1 : 0
  role       = aws_iam_role.codepipeline_service_role[0].name
  policy_arn = aws_iam_policy.codepipeline_service_policy[0].arn
}