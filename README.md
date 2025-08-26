# 

# 🚀 cybersecurity-threat-detection-platform (Lambda + SageMaker + XGBoost)

## 🔹 Project Overview
This project builds a comprehensive **production-ready Cybersecurity threat detection stack** on AWS with Lambda and SageMaker.  
It uses **XGboost** to train and test the model

This stack is deployed using Codebuild/CodePipeline via terraform

---

## ✨ Features
- **XGBoost Model**: Optimized for cybersecurity threat detection with 5 attack categories
- **Feature Engineering**: Advanced feature creation for network traffic analysis
- **Real-time & Batch Prediction**: Flexible inference options
- **Hyperparameter Tuning**: Automated optimization using SageMaker
- **Multi-class Classification**: Normal, DoS, Probe, R2L, and U2R attack detection
- **Scalable Architecture**: Cloud-native design with auto-scaling capabilities

---

## 🛠️ Tech Stack
- **Data Pipeline**: Processing security logs and network data
- **ML Pipeline**: XGBoost model training and evaluation using SageMaker
- **Infrastructure**: Terraform for AWS resource provisioning
- **CI/CD**: CodeBuild and CodePipeline for automated deployment
- **Model Deployment**: SageMaker endpoint for real-time threat detection-

---

## 📂 Project Structure
```bash
cyber-threat-detection/
├── README.md
├── .gitignore
├── requirements.txt
├── buildspec.yml
├── terraform/
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   ├── versions.tf
│   ├── modules/
│   │   ├── sagemaker/
│   │   │   ├── main.tf
│   │   │   ├── variables.tf
│   │   │   └── outputs.tf
│   │   ├── s3/
│   │   │   ├── main.tf
│   │   │   ├── variables.tf
│   │   │   └── outputs.tf
│   │   ├── iam/
│   │   │   ├── main.tf
│   │   │   ├── variables.tf
│   │   │   └── outputs.tf
│   │   └── codepipeline/
│   │       ├── main.tf
│   │       ├── variables.tf
│   │       └── outputs.tf
│   └── environments/
│       ├── dev/
│       │   ├── main.tf
│       │   ├── terraform.tfvars
│       │   └── backend.tf
│       └── prod/
│           ├── main.tf
│           ├── terraform.tfvars
│           └── backend.tf
├── src/
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── feature_engineering.py
│   │   └── data_validation.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── hyperparameter_tuning.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictor.py
│   │   └── batch_transform.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── helpers.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
│       └── kdd_cup_sample.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_development.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_data_processing.py
│   ├── test_training.py
│   └── test_inference.py
├── scripts/
│   ├── setup_environment.sh
│   ├── deploy_infrastructure.sh
│   └── run_pipeline.sh
└── docs/
    ├── architecture.md
    ├── deployment_guide.md
    └── api_reference.md


```
---

