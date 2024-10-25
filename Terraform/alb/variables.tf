# プロジェクト名
variable "project_name" {
  description = "Name of the project"
  type        = string
}

# VPC ID
variable "vpc_id" {
  description = "The ID of the VPC"
  type        = string
}

# パブリックサブネット
variable "public_subnets" {
  description = "The IDs of the public subnets"
  type        = list(string)
}

# ALBセキュリティグループ
variable "alb_sg_id" {
  description = "The ID of the ALB security group"
  type        = string
}
