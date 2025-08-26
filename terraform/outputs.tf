output "s3_bucket_data" {
  description = "S3 bucket for data storage"
  value       = aws_s3_bucket.data_bucket.bucket
}

output "s3_bucket_models" {
  description = "S3 bucket for model artifacts"
  value       = aws_s3_bucket.model_bucket.bucket
}

output "sagemaker_execution_role_arn" {
  description = "SageMaker execution role ARN"
  value       = aws_iam_role.sagemaker_execution_role.arn
}

output "codepipeline_name" {
  description = "CodePipeline name"
  value       = aws_codepipeline.ml_pipeline.name
}