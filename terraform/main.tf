terraform {
  required_version = ">= 1.0"
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

# Random suffix for unique resource names
resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}

# Local values for consistent naming
locals {
  name_prefix = "${var.project_name}-${var.environment}"
  common_tags = merge(var.common_tags, {
    Environment = var.environment
    Project     = var.project_name
    ManagedBy   = "Terraform"
    Module      = "cybersecurity-ml"
  })
}