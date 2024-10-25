# ALBの作成
resource "aws_lb" "main" {
  name               = "${var.project_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [var.alb_sg_id]  # 変更: セキュリティグループをモジュール入力から参照
  subnets            = var.public_subnets  # 変更: サブネットをモジュール入力から参照

  tags = {
    Name = "${var.project_name}-alb"
  }
}

# ALBリスナーの作成
resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}

# ターゲットグループの作成
resource "aws_lb_target_group" "app" {
  name        = "${var.project_name}-tg"
  port        = 8501
  protocol    = "HTTP"
  vpc_id      = var.vpc_id  # 変更: VPC IDをモジュール入力から参照
  target_type = "ip"

  health_check {
    healthy_threshold   = "3"
    interval            = "30"
    protocol            = "HTTP"
    matcher             = "200"
    timeout             = "3"
    path                = "/_stcore/health"
    unhealthy_threshold = "2"
  }

  tags = {
    Name = "${var.project_name}-tg"
  }
}
