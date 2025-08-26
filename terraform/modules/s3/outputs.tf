# S3 Module Outputs

output "data_bucket_name" {
  description = "Name of the data bucket"
  value       = aws_s3_bucket.data_bucket.bucket
}

output "data_bucket_arn" {
  description = "ARN of the data bucket"
  value       = aws_s3_bucket.data_bucket.arn
}

output "data_bucket_domain_name" {
  description = "Domain name of the data bucket"
  value       = aws_s3_bucket.data_bucket.bucket_domain_name
}

output "model_bucket_name" {
  description = "Name of the model artifacts bucket"
  value       = aws_s3_bucket.model_bucket.bucket
}

output "model_bucket_arn" {
  description = "ARN of the model artifacts bucket"
  value       = aws_s3_bucket.model_bucket.arn
}

output "model_bucket_domain_name" {
  description = "Domain name of the model artifacts bucket"
  value       = aws_s3_bucket.model_bucket.bucket_domain_name
}

output "pipeline_bucket_name" {
  description = "Name of the pipeline artifacts bucket"
  value       = var.create_pipeline_bucket ? aws_s3_bucket.pipeline_artifacts[0].bucket : null
}

output "pipeline_bucket_arn" {
  description = "ARN of the pipeline artifacts bucket"
  value       = var.create_pipeline_bucket ? aws_s3_bucket.pipeline_artifacts[0].arn : null
}

output "bucket_names" {
  description = "Map of all bucket names"
  value = {
    data_bucket     = aws_s3_bucket.data_bucket.bucket
    model_bucket    = aws_s3_bucket.model_bucket.bucket
    pipeline_bucket = var.create_pipeline_bucket ? aws_s3_bucket.pipeline_artifacts[0].bucket : null
  }
}

output "bucket_arns" {
  description = "Map of all bucket ARNs"
  value = {
    data_bucket     = aws_s3_bucket.data_bucket.arn
    model_bucket    = aws_s3_bucket.model_bucket.arn
    pipeline_bucket = var.create_pipeline_bucket ? aws_s3_bucket.pipeline_artifacts[0].arn : null
  }
}