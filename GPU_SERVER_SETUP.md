# PolySight GPU Server Setup Guide

GCP Compute Engine에서 GPU VM을 셋업하여 ColPali (Jina V4) + Docling OCR을 로컬 모드로 가속하는 가이드입니다.

## 1. VM 사양

| 항목 | 값 |
|------|------|
| **인스턴스 이름** | polysight-gpu |
| **리전/영역** | us-west1 (오리건) |
| **머신 타입** | g2-standard-8 (8 vCPU, 32GB RAM) |
| **GPU** | NVIDIA L4 x 1 (24GB VRAM) |
| **OS 이미지** | Deep Learning VM with CUDA 12.4 M129 (Debian 11, Python 3.10) |
| **디스크** | 100GB Balanced Persistent Disk |
| **네트워크 태그** | http-server, https-server, polysight |
| **예상 비용** | ~$633/월 (on-demand), ~$0.87/시간 |

## 2. 방화벽 규칙

| 규칙 이름 | 방향 | 대상 태그 | 소스 | 프로토콜/포트 |
|-----------|------|-----------|------|--------------|
| allow-polysight-gradio | 인그레스 | polysight | 0.0.0.0/0 | TCP 7860 |
| default-allow-http | 인그레스 | http-server | 0.0.0.0/0 | TCP 80 |
| default-allow-https | 인그레스 | https-server | 0.0.0.0/0 | TCP 443 |

## 3. 초기 셋업 (SSH 접속 후)

### 3.1 NVIDIA 드라이버 확인

```bash
nvidia-smi
# NVIDIA L4 24GB, CUDA 12.4 확인
```

### 3.2 프로젝트 클론

```bash
cd ~
git clone <your-polysight-repo-url> polysight
cd polysight
```

### 3.3 Python 환경 및 의존성

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.4 환경변수 설정

```bash
cp .env.example .env
nano .env
```

`.env` 파일에 아래 값을 입력합니다:

```bash
# Elasticsearch (필수)
ELASTIC_CLOUD_SERVERLESS_URL=https://psearch-xxxx.es.us-west-2.aws.elastic.cloud:443
ELASTIC_API_KEY=your-actual-api-key

# Jina V4 — GPU VM에서는 로컬 모드 사용 (API Key 비워두기)
JINA_API_KEY=

# HuggingFace — 모델 다운로드 시 필요할 수 있음
HF_TOKEN=your-hf-token-if-needed
```

> **보안 참고**: `.env`와 `config.json`은 `.gitignore`에 포함되어 있으므로 git에 push되지 않습니다.
> `config.json`은 UI에서 저장한 값으로, `.env`보다 높은 우선순위를 가집니다.

### 3.5 설정 우선순위

PolySight의 ConfigManager는 다음 순서로 설정을 읽습니다:

1. `config.json` (최우선 — UI Settings 탭에서 저장)
2. `.env` 파일
3. 기본값

GPU VM에서는 `.env`만 설정하면 충분합니다. Jina API Key를 비워두면 자동으로 로컬(GPU) 모드로 전환됩니다.

## 4. Gradio 앱 실행

```bash
cd ~/polysight
source .venv/bin/activate
python frontend/app.py
```

Gradio가 포트 7860에서 실행됩니다. 브라우저에서 접속:

```
http://<외부IP>:7860
```

현재 외부 IP: `http://34.19.5.214:7860`

### 백그라운드 실행 (nohup)

```bash
nohup python frontend/app.py > logs/gradio.log 2>&1 &
```

### systemd 서비스로 등록 (선택)

```bash
sudo tee /etc/systemd/system/polysight.service << 'EOF'
[Unit]
Description=PolySight Gradio App
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/polysight
Environment="PATH=/home/$USER/polysight/.venv/bin:/usr/local/bin:/usr/bin"
ExecStart=/home/$USER/polysight/.venv/bin/python frontend/app.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable polysight
sudo systemctl start polysight
```

## 5. GPU 성능 기대치

| 작업 | API 모드 (Jina Cloud) | 로컬 GPU (L4) |
|------|----------------------|---------------|
| Jina V4 임베딩 (페이지당) | ~0.5초 | ~0.1초 |
| Docling OCR (페이지당) | N/A (항상 로컬) | ~1-2초 |
| 10페이지 문서 인제스트 | ~8-10초 | ~3-5초 |

## 6. VM 관리 팁

### 비용 절약을 위한 중지/시작

```bash
# GCP Cloud Shell 또는 gcloud CLI
gcloud compute instances stop polysight-gpu --zone=us-west1-a
gcloud compute instances start polysight-gpu --zone=us-west1-a
```

> 중지하면 GPU/CPU 비용이 발생하지 않습니다 (디스크 비용만 유지).
> SUD (Sustained Use Discount): 월 내내 실행하면 자동 할인 적용.

### SSH 접속

```bash
# 브라우저에서
# GCP Console > Compute Engine > VM 인스턴스 > polysight-gpu > SSH

# 또는 gcloud CLI
gcloud compute ssh polysight-gpu --zone=us-west1-a
```

## 7. 트러블슈팅

| 문제 | 해결 |
|------|------|
| `nvidia-smi` 안됨 | `sudo /opt/deeplearning/install-driver.sh` |
| CUDA 버전 불일치 | PyTorch CUDA 버전 확인: `python -c "import torch; print(torch.cuda.is_available())"` |
| 포트 7860 접속 불가 | 방화벽 규칙 확인, VM 네트워크 태그에 `polysight` 포함 여부 확인 |
| OOM (메모리 부족) | batch_size 줄이기 또는 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 설정 |
| Elastic 연결 실패 | `.env`의 URL과 API Key 확인, `curl -k` 으로 직접 테스트 |

## 8. 동일 환경 재현 (다른 사람용)

1. GCP Console > Compute Engine > 인스턴스 만들기
2. 위 **1. VM 사양** 표대로 설정
3. 방화벽: HTTP/HTTPS 트래픽 허용 체크 + 네트워크 태그 `polysight` 추가
4. 방화벽 규칙 `allow-polysight-gradio` 만들기 (TCP 7860, 대상 태그 polysight)
5. SSH 접속 후 **3. 초기 셋업** 단계 따라 실행
