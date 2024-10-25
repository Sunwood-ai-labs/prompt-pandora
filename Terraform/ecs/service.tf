# ECSサービスの作成
resource "aws_ecs_service" "app" {
  name            = "${var.project_name}-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = var.app_count
  launch_type     = "FARGATE"

  network_configuration {
    security_groups  = [var.ecs_sg_id]  # 修正: セキュリティグループをモジュール変数から参照
    subnets         = var.public_subnets  # 修正: サブネットをモジュール変数から参照
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = var.target_group_arn  # 修正: ターゲットグループをモジュール変数から参照
    container_name   = "${var.project_name}-container"
    container_port   = 8501
  }

  depends_on = [aws_ecs_task_definition.app]  # 修正: 依存関係を修正
}
