# 営業時間の開始時刻
variable "business_hours_start" {
  description = "Start time of business hours (in UTC)"
  type        = string
  default     = "22:00"  # JST 07:00
}

# 営業時間の終了時刻
variable "business_hours_end" {
  description = "End time of business hours (in UTC)"
  type        = string
  default     = "11:00"  # JST 20:00
}

# Auto Scaling用のスケジュールルール
resource "aws_appautoscaling_scheduled_action" "scale_up" {
  count              = var.schedule_enabled ? 1 : 0
  name               = "${var.project_name}-scale-up"
  service_namespace  = "ecs"
  resource_id        = aws_appautoscaling_target.app_scale_target.resource_id
  scalable_dimension = aws_appautoscaling_target.app_scale_target.scalable_dimension
  schedule          = "cron(0 22 ? * MON-FRI *)"  # UTC 22:00 (JST 07:00)

  scalable_target_action {
    min_capacity = var.app_count
    max_capacity = var.app_count
  }
}

resource "aws_appautoscaling_scheduled_action" "scale_down" {
  count              = var.schedule_enabled ? 1 : 0
  name               = "${var.project_name}-scale-down"
  service_namespace  = "ecs"
  resource_id        = aws_appautoscaling_target.app_scale_target.resource_id
  scalable_dimension = aws_appautoscaling_target.app_scale_target.scalable_dimension
  schedule          = "cron(0 11 ? * MON-FRI *)"  # UTC 11:00 (JST 20:00)

  scalable_target_action {
    min_capacity = 0
    max_capacity = 0
  }
}

# Auto Scalingターゲットの設定
resource "aws_appautoscaling_target" "app_scale_target" {
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.app.name}"  # 修正: main → app
  scalable_dimension = "ecs:service:DesiredCount"
  min_capacity      = var.schedule_enabled ? 0 : var.app_count
  max_capacity      = var.app_count
}
