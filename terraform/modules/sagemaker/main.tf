resource "aws_sagemaker_notebook_instance" "example" {
  name          = "threat-detection-notebook"
  instance_type = "ml.t2.medium"
  role_arn      = var.role_arn
}
