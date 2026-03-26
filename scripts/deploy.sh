#!/bin/bash
# PolySight Deploy Script
# VM에서 실행: bash scripts/deploy.sh
#
# 사용법:
#   첫 배포:   bash scripts/deploy.sh --init
#   업데이트:   bash scripts/deploy.sh
#   상태 확인:  bash scripts/deploy.sh --status

set -e

# ============================================
# Configuration
# ============================================
APP_NAME="polysight"
APP_DIR="$HOME/polysight"
VENV_DIR="$APP_DIR/.venv"
SERVICE_NAME="polysight"
PORT=7860
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[PolySight]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ============================================
# Commands
# ============================================

show_status() {
    log "=== PolySight Status ==="
    echo ""

    # Service status
    if systemctl is-active --quiet $SERVICE_NAME 2>/dev/null; then
        echo -e "  Service:  ${GREEN}● Running${NC}"
    else
        echo -e "  Service:  ${RED}● Stopped${NC}"
    fi

    # GPU
    if nvidia-smi &>/dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1)
        GPU_MEM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
        echo "  GPU:      $GPU_NAME ($GPU_MEM MiB)"
    else
        echo -e "  GPU:      ${YELLOW}Not available${NC}"
    fi

    # Port
    if ss -tlnp | grep -q ":$PORT"; then
        echo -e "  Port:     ${GREEN}$PORT listening${NC}"
    else
        echo -e "  Port:     ${RED}$PORT not listening${NC}"
    fi

    # External IP
    EXTERNAL_IP=$(curl -s -m 5 http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip -H "Metadata-Flavor: Google" 2>/dev/null || echo "unknown")
    echo "  URL:      http://$EXTERNAL_IP:$PORT"

    # Git
    if [ -d "$APP_DIR/.git" ]; then
        cd "$APP_DIR"
        BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
        COMMIT=$(git log -1 --format='%h %s' 2>/dev/null || echo "unknown")
        echo "  Branch:   $BRANCH"
        echo "  Commit:   $COMMIT"
    fi

    # Disk
    DISK_USAGE=$(df -h "$APP_DIR" 2>/dev/null | tail -1 | awk '{print $3"/"$2" ("$5")"}')
    echo "  Disk:     $DISK_USAGE"

    echo ""

    # Recent logs
    if systemctl is-active --quiet $SERVICE_NAME 2>/dev/null; then
        log "Recent logs (last 10 lines):"
        journalctl -u $SERVICE_NAME --no-pager -n 10 2>/dev/null || true
    fi
}

init_deploy() {
    log "=== First-time Setup ==="

    # 1. Check GPU
    log "Checking GPU..."
    if nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    else
        warn "No GPU detected. Will use CPU mode."
    fi

    # 2. Create venv
    if [ ! -d "$VENV_DIR" ]; then
        log "Creating Python virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi

    # 3. Install dependencies
    log "Installing dependencies..."
    $PIP install --upgrade pip
    $PIP install -r "$APP_DIR/requirements.txt"

    # Install PyTorch with CUDA if GPU available
    if nvidia-smi &>/dev/null; then
        log "Installing PyTorch with CUDA support..."
        $PIP install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    fi

    # 4. Setup .env if not exists
    if [ ! -f "$APP_DIR/.env" ]; then
        if [ -f "$APP_DIR/.env.example" ]; then
            cp "$APP_DIR/.env.example" "$APP_DIR/.env"
            warn ".env created from .env.example — please edit with your API keys:"
            warn "  nano $APP_DIR/.env"
        fi
    fi

    # 5. Create log directory
    mkdir -p "$APP_DIR/logs"

    # 6. Install systemd service
    log "Installing systemd service..."
    ACTUAL_USER=$(whoami)
    sudo tee /etc/systemd/system/$SERVICE_NAME.service > /dev/null << SERVICEOF
[Unit]
Description=PolySight - Agent Battle Demo
After=network.target
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=simple
User=$ACTUAL_USER
WorkingDirectory=$APP_DIR
Environment="PATH=$VENV_DIR/bin:/usr/local/bin:/usr/bin"
Environment="PYTHONUNBUFFERED=1"
Environment="GRADIO_SERVER_NAME=0.0.0.0"
ExecStart=$PYTHON frontend/app.py
Restart=on-failure
RestartSec=15
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SERVICEOF

    sudo systemctl daemon-reload
    sudo systemctl enable $SERVICE_NAME

    log "✅ Init complete!"
    log ""
    log "Next steps:"
    log "  1. Edit .env:  nano $APP_DIR/.env"
    log "  2. Start app:  sudo systemctl start $SERVICE_NAME"
    log "  3. Check:      bash scripts/deploy.sh --status"
}

update_deploy() {
    log "=== Updating PolySight ==="

    cd "$APP_DIR"

    # 1. Pull latest code
    log "Pulling latest code..."
    git pull

    # 2. Update dependencies (only if requirements changed)
    if git diff HEAD~1 --name-only | grep -q "requirements.txt"; then
        log "requirements.txt changed — installing new dependencies..."
        $PIP install -r requirements.txt
    else
        log "No dependency changes."
    fi

    # 3. Restart service
    log "Restarting service..."
    sudo systemctl restart $SERVICE_NAME

    # 4. Wait and check
    sleep 3
    if systemctl is-active --quiet $SERVICE_NAME; then
        log "✅ Update complete! Service is running."
    else
        error "Service failed to start. Check: journalctl -u $SERVICE_NAME -n 50"
    fi

    show_status
}

# ============================================
# Main
# ============================================

case "${1:-}" in
    --init)
        init_deploy
        ;;
    --status)
        show_status
        ;;
    --logs)
        journalctl -u $SERVICE_NAME -f
        ;;
    --restart)
        log "Restarting service..."
        sudo systemctl restart $SERVICE_NAME
        sleep 2
        show_status
        ;;
    --stop)
        log "Stopping service..."
        sudo systemctl stop $SERVICE_NAME
        show_status
        ;;
    *)
        update_deploy
        ;;
esac
