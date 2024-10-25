# AWSリージョンの設定
variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "ap-northeast-1"
}

# プロジェクト名の設定
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "streamlit-app"
}

# VPCのネットワーク設定
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

# コンテナイメージの設定
variable "container_image" {
  description = "Container image to deploy"
  type        = string
}

# タスクのリソース設定
variable "task_cpu" {
  description = "CPU units for the task"
  type        = string
  default     = "256"
}

# タスクのメモリ設定
variable "task_memory" {
  description = "Memory (MiB) for the task"
  type        = string
  default     = "512"
}

# アプリケーションのインスタンス数
variable "app_count" {
  description = "Number of application instances to run"
  type        = number
  default     = 1
}
