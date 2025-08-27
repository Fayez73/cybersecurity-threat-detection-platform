# Backend configuration for Production environment
# Uncomment and configure the backend block below after creating the S3 bucket and DynamoDB table

# terraform {
#   backend "s3" {
#     bucket         = "your-terraform-state-bucket-prod"
#     key            = "cybersecurity-threat-detection/prod/terraform.tfstate"
#     region         = "us-east-1"
#     dynamodb_table = "terraform-state-locks-prod"
#     encrypt        = true
#   }
# }

# Instructions for setting up the production backend:
#
# IMPORTANT: Use separate state infrastructure for production!
#
# 1. Create a dedicated S3 bucket for production Terraform state:
#    aws s3 mb s3://your-terraform-state-bucket-prod --region us-east-1
#    aws s3api put-bucket-versioning --bucket your-terraform-state-bucket-prod --versioning-configuration Status=Enabled
#    aws s3api put-bucket-encryption --bucket your-terraform-state-bucket-prod --server-side-encryption-configuration '{
#      "Rules": [{
#        "ApplyServerSideEncryptionByDefault": {
#          "SSEAlgorithm": "AES256"
#        }
#      }]
#    }'
#    aws s3api put-public-access-block --bucket your-terraform-state-bucket-prod --public-access-block-configuration '{
#      "BlockPublicAcls": true,
#      "IgnorePublicAcls": true,
#      "BlockPublicPolicy": true,
#      "RestrictPublicBuckets": true
#    }'
#
# 2. Create a dedicated DynamoDB table for production state locking:
#    aws dynamodb create-table \
#      --table-name terraform-state-locks-prod \
#      --attribute-definitions AttributeName=LockID,AttributeType=S \
#      --key-schema AttributeName=LockID,KeyType=HASH \
#      --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5 \
#      --region us-east-1
#
# 3. Apply additional security to the production state bucket:
#    - Enable MFA delete
#    - Set up bucket notifications for audit trail
#    - Configure cross-region replication for disaster recovery
#
# 4. Uncomment the terraform block above and replace bucket names with your actual names
#
# 5. Initialize the backend:
#    terraform init
#
# Best Practices for Production:
# - Use separate AWS accounts for prod/non-prod
# - Enable AWS CloudTrail for audit logging
# - Use AWS Config for compliance monitoring
# - Implement proper IAM policies with least privilege
# - Set up monitoring and alerting for state file access