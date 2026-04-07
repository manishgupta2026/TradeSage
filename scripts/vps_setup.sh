#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
#  TradeSage VPS Setup Script
#  Full Ubuntu 22.04 deployment: Docker + Nginx + HTTPS
# ═══════════════════════════════════════════════════════════
#
# RECOMMENDED VPS SPECS
# ─────────────────────────────────────────
# MINIMUM (scan only, no retraining):
#   2 vCPU | 4GB RAM | 40GB SSD
#   ~$12-18/month → Hetzner CX22
#
# RECOMMENDED (scan + daily retrain 3000+ stocks):
#   4 vCPU | 8GB RAM | 80GB SSD
#   ~$24-28/month → Hetzner CX32
#   (Retraining XGBoost on 3000 stocks is CPU-heavy)
#
# PROVIDER: Hetzner (best price/perf)
# REGION: Hetzner Helsinki or Nuremberg
#   (DigitalOcean Bangalore if latency to NSE is priority)
# OS: Ubuntu 22.04 LTS
# ─────────────────────────────────────────
#
# Usage:
#   chmod +x scripts/vps_setup.sh
#   sudo ./scripts/vps_setup.sh
#

set -euo pipefail

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }
step() { echo -e "\n${CYAN}═══ $1 ═══${NC}"; }

# ── Check root ──
if [ "$EUID" -ne 0 ]; then
    err "Please run as root: sudo ./scripts/vps_setup.sh"
fi

# ── Prompt for config ──
step "TRADESAGE VPS SETUP"
echo ""
read -p "Enter your domain name (e.g., tradesage.example.com): " DOMAIN
read -p "Enter your Git repo URL: " REPO_URL
REPO_URL=${REPO_URL:-"https://github.com/manishgupta2026/TradeSage.git"}
read -p "Deploy directory [/opt/tradesage]: " DEPLOY_DIR
DEPLOY_DIR=${DEPLOY_DIR:-"/opt/tradesage"}

echo ""
log "Domain:     $DOMAIN"
log "Repo:       $REPO_URL"
log "Deploy dir: $DEPLOY_DIR"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
[[ ! $REPLY =~ ^[Yy]$ ]] && exit 0

# ══════════════════════════════════════════════════════════
#  1. System Update
# ══════════════════════════════════════════════════════════
step "1/7  System Update"
apt-get update -y && apt-get upgrade -y
apt-get install -y curl git wget unzip htop ncdu jq
log "System updated"

# ══════════════════════════════════════════════════════════
#  2. Install Docker + Docker Compose
# ══════════════════════════════════════════════════════════
step "2/7  Installing Docker"

if command -v docker &> /dev/null; then
    log "Docker already installed: $(docker --version)"
else
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
    log "Docker installed: $(docker --version)"
fi

# Docker Compose (plugin)
if docker compose version &> /dev/null; then
    log "Docker Compose already available"
else
    apt-get install -y docker-compose-plugin
    log "Docker Compose installed"
fi

# ══════════════════════════════════════════════════════════
#  3. Install Nginx
# ══════════════════════════════════════════════════════════
step "3/7  Installing Nginx"

apt-get install -y nginx
systemctl enable nginx

# Configure Nginx
cat > /etc/nginx/sites-available/tradesage << 'NGINX_EOF'
limit_req_zone $binary_remote_addr zone=tradesage_ratelimit:10m rate=100r/m;

upstream tradesage_backend {
    server 127.0.0.1:8000;
    keepalive 32;
}

server {
    listen 80;
    server_name DOMAIN_PLACEHOLDER;

    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_min_length 256;
    gzip_types text/plain text/css text/javascript application/javascript application/json application/xml text/xml image/svg+xml;

    limit_req zone=tradesage_ratelimit burst=20 nodelay;
    limit_req_status 429;

    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    location / {
        proxy_pass http://tradesage_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 10s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;
    }

    location /api/signals/live {
        proxy_pass http://tradesage_backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding off;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
        limit_req off;
    }

    location /api/status {
        proxy_pass http://tradesage_backend;
        limit_req off;
    }
}
NGINX_EOF

# Replace domain placeholder
sed -i "s/DOMAIN_PLACEHOLDER/$DOMAIN/g" /etc/nginx/sites-available/tradesage

# Enable site
ln -sf /etc/nginx/sites-available/tradesage /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

nginx -t && systemctl reload nginx
log "Nginx configured for $DOMAIN"

# ══════════════════════════════════════════════════════════
#  4. Install Certbot (Let's Encrypt HTTPS)
# ══════════════════════════════════════════════════════════
step "4/7  Setting up HTTPS (Let's Encrypt)"

apt-get install -y certbot python3-certbot-nginx

echo ""
warn "Certbot will now request an SSL certificate for: $DOMAIN"
warn "Make sure your domain's DNS A record points to this server's IP!"
echo ""
read -p "Request SSL certificate now? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos --email admin@"$DOMAIN" || {
        warn "Certbot failed — you can retry later: sudo certbot --nginx -d $DOMAIN"
    }
    log "SSL certificate installed"
else
    warn "Skipped SSL. Run later: sudo certbot --nginx -d $DOMAIN"
fi

# ══════════════════════════════════════════════════════════
#  5. Clone repo & configure
# ══════════════════════════════════════════════════════════
step "5/7  Cloning TradeSage"

mkdir -p "$DEPLOY_DIR"
if [ -d "$DEPLOY_DIR/.git" ]; then
    cd "$DEPLOY_DIR"
    git pull
    log "Repository updated"
else
    git clone "$REPO_URL" "$DEPLOY_DIR"
    log "Repository cloned"
fi

cd "$DEPLOY_DIR"

# Create .env from template
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        warn "Created .env from .env.example — EDIT IT NOW with your credentials!"
        echo ""
        echo "  nano $DEPLOY_DIR/.env"
        echo ""
        read -p "Press Enter after editing .env, or Ctrl+C to abort..."
    else
        err ".env.example not found!"
    fi
fi

# Create required directories
mkdir -p data_cache_angel models logs data config frontend

log "Project configured"

# ══════════════════════════════════════════════════════════
#  6. UFW Firewall
# ══════════════════════════════════════════════════════════
step "6/7  Configuring Firewall (UFW)"

apt-get install -y ufw
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS

# Enable UFW non-interactively
echo "y" | ufw enable
ufw status
log "Firewall configured (22, 80, 443 only)"

# ══════════════════════════════════════════════════════════
#  7. Launch with Docker Compose
# ══════════════════════════════════════════════════════════
step "7/7  Starting TradeSage"

cd "$DEPLOY_DIR"
docker compose build --no-cache
docker compose up -d

# Wait for services
sleep 5
docker compose ps

echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""
log "🚀 TradeSage is now LIVE!"
echo ""
echo "  Dashboard:  http://$DOMAIN"
echo "  API Status: http://$DOMAIN/api/status"
echo "  Logs:       docker compose -f $DEPLOY_DIR/docker-compose.yml logs -f"
echo ""
echo "  Useful commands:"
echo "    docker compose logs -f scanner     # Watch scanner output"
echo "    docker compose logs -f retrainer   # Watch retrainer output"
echo "    docker compose restart scanner     # Restart scanner"
echo "    docker compose down                # Stop all services"
echo ""
echo "═══════════════════════════════════════════════════════════"
