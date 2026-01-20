#!/bin/bash
# Spreadsheet Normalizer 健康检查脚本
# 定期检查服务状态，自动重启失败的服务

# 日志文件
LOG_FILE="/var/log/spreadsheet-normalizer/healthcheck.log"

# 创建日志目录
mkdir -p "$(dirname "$LOG_FILE")"

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 检查后端服务
check_backend() {
    if ! systemctl is-active --quiet spreadsheet-normalizer-backend; then
        log "❌ Backend service is down! Attempting to restart..."
        systemctl restart spreadsheet-normalizer-backend
        sleep 5

        if systemctl is-active --quiet spreadsheet-normalizer-backend; then
            log "✅ Backend service restarted successfully"
        else
            log "❌ Failed to restart backend service"
            return 1
        fi
    fi
    return 0
}

# 检查 Nginx
check_nginx() {
    if ! systemctl is-active --quiet nginx; then
        log "❌ Nginx is down! Attempting to restart..."
        systemctl restart nginx
        sleep 3

        if systemctl is-active --quiet nginx; then
            log "✅ Nginx restarted successfully"
        else
            log "❌ Failed to restart Nginx"
            return 1
        fi
    fi
    return 0
}

# 检查 API 健康端点
check_api_health() {
    local health_url="http://127.0.0.1:8000/health"

    if ! curl -f -s "$health_url" > /dev/null 2>&1; then
        log "❌ API health check failed at $health_url"
        log "   Attempting to restart backend service..."
        systemctl restart spreadsheet-normalizer-backend
        sleep 5

        if curl -f -s "$health_url" > /dev/null 2>&1; then
            log "✅ API health check passed after restart"
        else
            log "❌ API still unhealthy after restart"
            return 1
        fi
    fi
    return 0
}

# 检查磁盘空间
check_disk_space() {
    local threshold=90  # 磁盘使用率阈值（百分比）
    local usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')

    if [ "$usage" -gt "$threshold" ]; then
        log "⚠️  Disk usage is high: ${usage}%"
        log "   Consider running cleanup script"
    fi
}

# 检查内存使用
check_memory() {
    local threshold=90  # 内存使用率阈值（百分比）
    local usage=$(free | awk '/Mem:/ {printf "%.0f", $3/$2 * 100}')

    if [ "$usage" -gt "$threshold" ]; then
        log "⚠️  Memory usage is high: ${usage}%"
    fi
}

# 主检查流程
main() {
    log "🔍 Starting health check..."

    local all_ok=true

    # 检查后端服务
    if ! check_backend; then
        all_ok=false
    fi

    # 检查 Nginx
    if ! check_nginx; then
        all_ok=false
    fi

    # 检查 API 健康端点
    if ! check_api_health; then
        all_ok=false
    fi

    # 检查系统资源
    check_disk_space
    check_memory

    if [ "$all_ok" = true ]; then
        # 成功时不输出日志（避免日志过大）
        # log "✅ All services are healthy"
        :
    else
        log "❌ Some services are unhealthy"
    fi
}

# 执行主函数
main
