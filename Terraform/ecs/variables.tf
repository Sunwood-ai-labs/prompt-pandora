# プロジェクト名
variable "project_name" {
  description = "Name of the project"
  type        = string
}

# AWSリージョン
variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
}

# コンテナイメージ
variable "container_image" {
  description = "Container image to deploy"
  type        = string
}

# タスクのCPU
variable "task_cpu" {
  description = "CPU units for the task"
  type        = string
}

# タスクのメモリ
variable "task_memory" {
  description = "Memory (MiB) for the task"
  type        = string
}

# アプリケーション数
variable "app_count" {
  description = "Number of application instances to run"
  type        = number
}

# スケジューリング設定
variable "schedule_enabled" {
  description = "Enable scheduled scaling of the application"
  type        = bool
  default     = true
}

# 他のモジュールからの参照用変数
variable "vpc_id" {
  description = "The ID of the VPC"
  type        = string
}

variable "public_subnets" {
  description = "The IDs of the public subnets"
  type        = list(string)
}

variable "ecs_sg_id" {
  description = "The ID of the ECS tasks security group"
  type        = string
}

variable "execution_role_arn" {
  description = "The ARN of the ECS execution role"
  type        = string
}

variable "task_role_arn" {
  description = "The ARN of the ECS task role"
  type        = string
}

variable "target_group_arn" {
  description = "The ARN of the target group"
  type        = string
}
