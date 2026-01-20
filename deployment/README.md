# Deployment 部署配置和脚本

本目录包含 Spreadsheet Normalizer 应用的部署配置文件和脚本。

## 目录结构

```
deployment/
├── nginx/
│   └── spreadsheet-normalizer.conf   # Nginx 反向代理配置
├── systemd/
│   └── spreadsheet-normalizer-backend.service  # Systemd 服务配置
├── scripts/
│   ├── deploy.sh                    # 快速部署/更新脚本
│   └── healthcheck.sh               # 健康检查脚本
└── README.md                         # 本文件
```

## 配置文件使用

### Nginx 配置

**文件**: `nginx/spreadsheet-normalizer.conf`

**安装步骤**:
1. 复制到 Nginx 配置目录:
   ```bash
   sudo cp deployment/nginx/spreadsheet-normalizer.conf \
           /etc/nginx/sites-available/spreadsheet-normalizer
   ```

2. 编辑配置文件，修改 `server_name`:
   ```bash
   sudo nano /etc/nginx/sites-available/spreadsheet-normalizer
   ```

3. 启用配置:
   ```bash
   sudo ln -s /etc/nginx/sites-available/spreadsheet-normalizer \
              /etc/nginx/sites-enabled/
   ```

4. 测试并重新加载:
   ```bash
   sudo nginx -t
   sudo systemctl reload nginx
   ```

**配置说明**:
- 反向代理到 FastAPI 后端（127.0.0.1:8000）
- 提供前端静态文件（`/opt/spreadsheet-normalizer/frontend/dist`）
- 支持 WebSocket 连接
- 文件上传限制 100MB
- 静态资源缓存优化

### Systemd 服务配置

**文件**: `systemd/spreadsheet-normalizer-backend.service`

**安装步骤**:
1. 创建日志目录:
   ```bash
   sudo mkdir -p /var/log/spreadsheet-normalizer
   sudo chown $USER:$USER /var/log/spreadsheet-normalizer
   ```

2. 复制服务文件:
   ```bash
   sudo cp deployment/systemd/spreadsheet-normalizer-backend.service \
           /etc/systemd/system/
   ```

3. 如果用户不是 `ubuntu`，编辑服务文件修改 `User` 和 `Group`:
   ```bash
   sudo nano /etc/systemd/system/spreadsheet-normalizer-backend.service
   ```

4. 启动服务:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl start spreadsheet-normalizer-backend
   sudo systemctl enable spreadsheet-normalizer-backend
   ```

**配置说明**:
- Uvicorn 运行 FastAPI 应用
- 2 个 worker 进程（可根据 CPU 核心数调整）
- 监听 127.0.0.1:8000（仅本地访问）
- 自动重启策略
- 内存限制 2GB
- 日志输出到 `/var/log/spreadsheet-normalizer/`

## 脚本使用

### 1. 部署/更新脚本 (`deploy.sh`)

**用途**: 快速更新代码并重启服务

**功能**:
- 从 Git 拉取最新代码
- 更新 Python 依赖
- 重新构建前端
- 重启后端服务
- 重新加载 Nginx
- 显示服务状态

**使用方法**:
```bash
# 赋予执行权限（首次使用）
chmod +x deployment/scripts/deploy.sh

# 执行部署
./deployment/scripts/deploy.sh
```

**注意事项**:
- 需要 sudo 权限（用于重启服务）
- 确保已配置 Git 访问权限
- 会自动检查服务状态，失败时退出

**适用场景**:
- 代码更新后快速部署
- 修复 bug 后重新部署
- 添加新功能后发布

### 2. 健康检查脚本 (`healthcheck.sh`)

**用途**: 定期检查服务健康状态，自动重启失败的服务

**功能**:
- 检查后端服务状态
- 检查 Nginx 状态
- 检查 API 健康端点（`/health`）
- 检查磁盘和内存使用情况
- 自动重启失败的服务
- 记录日志到 `/var/log/spreadsheet-normalizer/healthcheck.log`

**使用方法**:

**手动执行**:
```bash
# 赋予执行权限（首次使用）
chmod +x deployment/scripts/healthcheck.sh

# 手动运行
sudo ./deployment/scripts/healthcheck.sh
```

**定时执行（推荐）**:
```bash
# 添加到 crontab（每 5 分钟检查一次）
sudo crontab -e

# 添加以下行:
*/5 * * * * /opt/spreadsheet-normalizer/deployment/scripts/healthcheck.sh
```

**日志查看**:
```bash
tail -f /var/log/spreadsheet-normalizer/healthcheck.log
```

**适用场景**:
- 生产环境自动监控
- 防止服务意外停止
- 资源使用监控

## 完整部署流程

### 首次部署

按照以下顺序执行:

1. **准备服务器环境**（参考 `../DEPLOYMENT.md`）
2. **上传代码到服务器**
3. **配置环境变量**:
   - 后端: `cp backend/.env.example backend/.env` 并编辑
   - 前端: `cp frontend/.env.production.example frontend/.env.production` 并编辑
4. **安装依赖**
5. **配置 Nginx**（使用 `nginx/spreadsheet-normalizer.conf`）
6. **配置 Systemd**（使用 `systemd/spreadsheet-normalizer-backend.service`）
7. **启动服务**
8. **配置健康检查**（设置 cron job）

详细步骤请参考项目根目录的 `DEPLOYMENT.md`。

### 更新部署

代码更新后:
```bash
cd /opt/spreadsheet-normalizer
./deployment/scripts/deploy.sh
```

## 监控和维护

### 查看服务状态

```bash
# 后端服务
sudo systemctl status spreadsheet-normalizer-backend

# Nginx
sudo systemctl status nginx
```

### 查看日志

```bash
# 后端实时日志
sudo journalctl -u spreadsheet-normalizer-backend -f

# 后端应用日志
tail -f /var/log/spreadsheet-normalizer/backend.log

# Nginx 访问日志
tail -f /var/log/nginx/spreadsheet-normalizer.access.log

# 健康检查日志
tail -f /var/log/spreadsheet-normalizer/healthcheck.log
```

### 重启服务

```bash
# 重启后端
sudo systemctl restart spreadsheet-normalizer-backend

# 重新加载 Nginx（不中断连接）
sudo systemctl reload nginx
```

## 故障排查

### 服务启动失败

```bash
# 查看详细错误信息
sudo journalctl -u spreadsheet-normalizer-backend -xe

# 检查配置文件
sudo systemctl cat spreadsheet-normalizer-backend

# 验证路径和权限
ls -la /opt/spreadsheet-normalizer/backend/venv/bin/uvicorn
```

### Nginx 配置错误

```bash
# 测试配置
sudo nginx -t

# 查看错误日志
sudo tail -f /var/log/nginx/error.log
```

### WebSocket 连接问题

```bash
# 检查 Nginx WebSocket 配置
sudo grep -A 10 "location /ws/" /etc/nginx/sites-available/spreadsheet-normalizer

# 检查后端日志
sudo journalctl -u spreadsheet-normalizer-backend -f
```

## 安全建议

1. **保护配置文件**:
   ```bash
   chmod 600 /opt/spreadsheet-normalizer/backend/.env
   ```

2. **限制脚本执行权限**:
   ```bash
   chmod 700 deployment/scripts/*.sh
   ```

3. **定期审查日志**:
   ```bash
   # 检查异常访问
   sudo grep "POST /api/v1/upload" /var/log/nginx/spreadsheet-normalizer.access.log
   ```

4. **启用防火墙**:
   ```bash
   sudo ufw status
   # 只开放 22, 80, 443 端口
   ```

## 额外资源

- **完整部署指南**: `../DEPLOYMENT.md`
- **项目文档**: `../CLAUDE.md`
- **快速开始**: `../QUICKSTART.md`

## 联系方式

- GitHub: https://github.com/ianwu0907/FYP2025_Ian
- Issues: https://github.com/ianwu0907/FYP2025_Ian/issues
