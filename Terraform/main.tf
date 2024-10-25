# このファイルは、StreamlitアプリケーションをAWS Fargateにデプロイするための
# モジュールを読み込みます。

# ネットワークモジュール
module "network" {
  source = "./network"

  project_name = var.project_name
  vpc_cidr     = var.vpc_cidr
}

# IAMモジュール
module "iam" {
  source = "./iam"

  project_name = var.project_name
}

# ECSモジュール
module "ecs" {
  source = "./ecs"

  project_name      = var.project_name
  aws_region        = var.aws_region
  container_image   = var.container_image
  task_cpu         = var.task_cpu
  task_memory      = var.task_memory
  app_count        = var.app_count
  schedule_enabled = true  # スケジューリングを有効化

  # 他のモジュールからの参照
  vpc_id           = module.network.vpc_id
  public_subnets   = module.network.public_subnet_ids
  ecs_sg_id        = module.network.ecs_sg_id
  execution_role_arn = module.iam.execution_role_arn
  task_role_arn     = module.iam.task_role_arn
  target_group_arn  = module.alb.target_group_arn

  depends_on = [module.network, module.iam, module.alb]
}

# ALBモジュール
module "alb" {
  source = "./alb"

  project_name    = var.project_name
  vpc_id         = module.network.vpc_id
  public_subnets = module.network.public_subnet_ids
  alb_sg_id      = module.network.alb_sg_id

  depends_on = [module.network]
}

# モニタリングモジュール
module "monitoring" {
  source = "./monitoring"

  project_name = var.project_name
}
