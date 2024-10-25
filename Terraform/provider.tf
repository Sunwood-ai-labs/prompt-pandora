# AWSプロバイダーの設定
provider "aws" {
  region = var.aws_region
}

# 利用可能なアベイラビリティーゾーンの取得
data "aws_availability_zones" "available" {
  state = "available"
}
