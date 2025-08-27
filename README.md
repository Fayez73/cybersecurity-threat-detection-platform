# 🚀 Cybersecurity-Threat-Detection-platform (Lambda + SageMaker + XGBoost)

## 🔹 Project Overview
This project builds a comprehensive A machine learning-based cybersecurity threat detection system that uses Amazon SageMaker, XGBoost, and deployed using Terraform and AWS CodePipeline.

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

## Quick Start
1. Clone the repository
2. Configure AWS credentials
3. Run `./scripts/setup_environment.sh`
4. Deploy infrastructure: `./scripts/deploy_infrastructure.sh`
5. Trigger the ML pipeline through CodePipeline

---

## Architecture
The system consists of:
- Data processing pipeline for cybersecurity logs
- XGBoost model training and evaluation
- Real-time inference endpoint
- Automated model retraining and deployment

---

## Documentation
See `/docs` directory for detailed documentation.

---

## 📂 Project Structure
```bash
cyber-threat-detection/
├── README.md
├── .gitignore
├── requirements.txt
├── buildspec.yml
├── data
│   └── sample
├── docs
├── notebooks
├── src
│   ├── data_processing
│   ├── inference
│   ├── training
│   └── utils
├── terraform
│   ├── envs
│   │   └── prod
│   └── modules
│       ├── codebuild
│       ├── codepipeline
│       ├── iam
│       ├── s3
│       └── sagemaker
|── scripts
└── tests

```
---

