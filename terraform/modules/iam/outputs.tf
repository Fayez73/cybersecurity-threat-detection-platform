# IAM Module Outputs

output "sagemaker_execution_role_arn" {
  description = "ARN of the SageMaker execution role"
  value       = aws_iam_role.sagemaker_execution_role.arn
}

output "sagemaker_execution_role_name" {
  description = "Name of the SageMaker execution role"
  value       = aws_iam_role.sagemaker_execution_role.name
}

output "sagemaker_execution_role_id" {
  description = "Unique ID of the SageMaker execution role"
  value       = aws_iam_role.sagemaker_execution_role.id
}

output "codebuild_service_role_arn" {
  description = "ARN of the CodeBuild service role"
  value       = var.create_codebuild_role ? aws_iam_role.codebuild_service_role[0].arn : null
}

output "codebuild_service_role_name" {
  description = "Name of the CodeBuild service role"
  value       = var.create_codebuild_role ? aws_iam_role.codebuild_service_role[0].name : null
}

output "codepipeline_service_role_arn" {
  description = "ARN of the CodePipeline service role"
  value       = var.create_codepipeline_role ? aws_iam_role.codepipeline_service_role[0].arn : null
}

output "codepipeline_service_role_name" {
  description = "Name of the CodePipeline service role"
  value       = var.create_codepipeline_role ? aws_iam_role.codepipeline_service_role[0].name : null
}

output "role_arns" {
  description = "Map of all IAM role ARNs"
  value = {
    sagemaker_execution  = aws_iam_role.sagemaker_execution_role.arn
    codebuild_service   = var.create_codebuild_role ? aws_iam_role.codebuild_service_role[0].arn : null
    codepipeline_service = var.create_codepipeline_role ? aws_iam_role.codepipeline_service_role[0].arn : null
  }
}

output "role_names" {
  description = "Map of all IAM role names"
  value = {
    sagemaker_execution  = aws_iam_role.sagemaker_execution_role.name
    codebuild_service   = var.create_codebuild_role ? aws_iam_role.codebuild_service_role[0].name : null
    codepipeline_service = var.create_codepipeline_role ? aws_iam_role.codepipeline_service_role[0].name : null
  }
}