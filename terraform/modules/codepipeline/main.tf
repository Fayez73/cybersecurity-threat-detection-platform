resource "aws_codepipeline" "pipeline" {
  name     = "${var.prefix}-pipeline"
  role_arn = var.role_arn
  artifact_store {
    location = var.bucket
    type     = "S3"
  }
  stage {
    name = "Source"
    action {
      name             = "SourceAction"
      category         = "Source"
      owner            = "AWS"
      provider         = "CodeCommit"
      version          = "1"
      output_artifacts = ["source_output"]
      configuration = {
        RepositoryName = var.repo_name
        BranchName     = "main"
      }
    }
  }

  stage {
    name = "Build"
    action {
      name             = "BuildAction"
      category         = "Build"
      owner            = "AWS"
      provider         = "CodeBuild"
      input_artifacts  = ["source_output"]
      output_artifacts = ["build_output"]
      version          = "1"
      configuration = {
        ProjectName = "example-build-project"
      }
    }
  }
}
