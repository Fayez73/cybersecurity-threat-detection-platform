# S3 Module for Cybersecurity Threat Detection System

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.4"
    }
  }
}

# Random suffix for unique bucket names
resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}

# Data bucket for training/test data
resource "aws_s3_bucket" "data_bucket" {
  bucket = "${var.project_name}-${var.environment}-data-${random_string.suffix.result}"
  tags   = var.tags
}

resource "aws_s3_bucket_versioning" "data_bucket_versioning" {
  count  = var.enable_versioning ? 1 : 0
  bucket = aws_s3_bucket.data_bucket.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data_bucket_encryption" {
  count  = var.enable_encryption ? 1 : 0
  bucket = aws_s3_bucket.data_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = var.kms_key_id != null ? "aws:kms" : "AES256"
      kms_master_key_id = var.kms_key_id
    }
    bucket_key_enabled = var.kms_key_id != null ? true : false
  }
}

resource "aws_s3_bucket_public_access_block" "data_bucket_pab" {
  bucket = aws_s3_bucket.data_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "data_bucket_lifecycle" {
  count  = var.enable_lifecycle ? 1 : 0
  bucket = aws_s3_bucket.data_bucket.id

  rule {
    id     = "training_data_lifecycle"
    status = "Enabled"

    filter {
      prefix = var.training_data_prefix
    }

    transition {
      days          = var.lifecycle_transition_days
      storage_class = "STANDARD_INFREQUENT_ACCESS"
    }

    transition {
      days          = var.lifecycle_transition_days * 2
      storage_class = "GLACIER"
    }

    expiration {
      days = var.data_retention_days
    }
  }

  rule {
    id     = "incomplete_multipart_uploads"
    status = "Enabled"

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# Model artifacts bucket
resource "aws_s3_bucket" "model_bucket" {
  bucket = "${var.project_name}-${var.environment}-models-${random_string.suffix.result}"
  tags   = var.tags
}

resource "aws_s3_bucket_versioning" "model_bucket_versioning" {
  count  = var.enable_versioning ? 1 : 0
  bucket = aws_s3_bucket.model_bucket.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "model_bucket_encryption" {
  count  = var.enable_encryption ? 1 : 0
  bucket = aws_s3_bucket.model_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = var.kms_key_id != null ? "aws:kms" : "AES256"
      kms_master_key_id = var.kms_key_id
    }
    bucket_key_enabled = var.kms_key_id != null ? true : false
  }
}

resource "aws_s3_bucket_public_access_block" "model_bucket_pab" {
  bucket = aws_s3_bucket.model_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "model_bucket_lifecycle" {
  count  = var.enable_lifecycle ? 1 : 0
  bucket = aws_s3_bucket.model_bucket.id

  rule {
    id     = "model_artifacts_lifecycle"
    status = "Enabled"

    filter {
      prefix = var.model_artifacts_prefix
    }

    transition {
      days          = var.lifecycle_transition_days * 2
      storage_class = "STANDARD_INFREQUENT_ACCESS"
    }

    expiration {
      days = var.model_retention_days
    }
  }
}

# Pipeline artifacts bucket (conditional)
resource "aws_s3_bucket" "pipeline_artifacts" {
  count  = var.create_pipeline_bucket ? 1 : 0
  bucket = "${var.project_name}-${var.environment}-pipeline-${random_string.suffix.result}"
  tags   = var.tags
}

resource "aws_s3_bucket_versioning" "pipeline_artifacts_versioning" {
  count  = var.create_pipeline_bucket && var.enable_versioning ? 1 : 0
  bucket = aws_s3_bucket.pipeline_artifacts[0].id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "pipeline_artifacts_encryption" {
  count  = var.create_pipeline_bucket && var.enable_encryption ? 1 : 0
  bucket = aws_s3_bucket.pipeline_artifacts[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "pipeline_artifacts_pab" {
  count  = var.create_pipeline_bucket ? 1 : 0
  bucket = aws_s3_bucket.pipeline_artifacts[0].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# S3 Bucket notifications (optional)
resource "aws_s3_bucket_notification" "data_bucket_notification" {
  count      = var.sns_topic_arn != null ? 1 : 0
  bucket     = aws_s3_bucket.data_bucket.id
  depends_on = [aws_s3_bucket_public_access_block.data_bucket_pab]

  topic {
    topic_arn = var.sns_topic_arn
    events    = ["s3:ObjectCreated:*"]
    filter_prefix = var.training_data_prefix
  }
}