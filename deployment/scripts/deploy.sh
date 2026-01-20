#!/bin/bash
# Spreadsheet Normalizer 部署脚本
# 用于快速更新代码和重启服务

set -e  # 遇到错误立即退出

echo "🚀 Starting deployment..."

# 配置变量
APP_DIR="/opt/spreadsheet-normalizer"
BACKEND_DIR="$APP_DIR/backend"
FRONTEND_DIR="$APP_DIR/frontend"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查是否在正确的目录
if [ ! -d "$APP_DIR" ]; then
    echo -e "${RED}❌ Error: $APP_DIR does not exist${NC}"
    exit 1
fi

cd $APP_DIR

# 1. 拉取最新代码
echo -e "${YELLOW}📥 Pulling latest code from Git...${NC}"
git pull origin main || {
    echo -e "${RED}❌ Git pull failed${NC}"
    exit 1
}

# 2. 更新后端依赖
echo -e "${YELLOW}🐍 Updating backend dependencies...${NC}"
cd $BACKEND_DIR
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. 安装 spreadsheet-normalizer 依赖
cd $APP_DIR/spreadsheet-normalizer
pip install -r requirements.txt

# 4. 重启后端服务
echo -e "${YELLOW}🔄 Restarting backend service...${NC}"
sudo systemctl restart spreadsheet-normalizer-backend

# 检查服务状态
if sudo systemctl is-active --quiet spreadsheet-normalizer-backend; then
    echo -e "${GREEN}✅ Backend service restarted successfully${NC}"
else
    echo -e "${RED}❌ Backend service failed to start${NC}"
    sudo systemctl status spreadsheet-normalizer-backend
    exit 1
fi

# 5. 更新前端
echo -e "${YELLOW}⚛️  Updating frontend...${NC}"
cd $FRONTEND_DIR
npm install
npm run build

# 6. 重新加载 Nginx
echo -e "${YELLOW}🔄 Reloading Nginx...${NC}"
sudo systemctl reload nginx

if sudo systemctl is-active --quiet nginx; then
    echo -e "${GREEN}✅ Nginx reloaded successfully${NC}"
else
    echo -e "${RED}❌ Nginx failed to reload${NC}"
    exit 1
fi

# 7. 显示服务状态
echo ""
echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
echo ""
echo "Service Status:"
echo "---------------"
sudo systemctl status spreadsheet-normalizer-backend --no-pager | head -n 5
echo ""
sudo systemctl status nginx --no-pager | head -n 3
echo ""
echo -e "${YELLOW}📊 Recent backend logs:${NC}"
sudo journalctl -u spreadsheet-normalizer-backend -n 10 --no-pager

echo ""
echo -e "${GREEN}🎉 All done!${NC}"
