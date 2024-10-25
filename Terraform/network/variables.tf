# プロジェクト名
variable "project_name" {
  description = "Name of the project"
  type        = string
}

# VPCのCIDR
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
}
