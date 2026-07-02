#!/bin/bash
# Install and enable a persistent user service for the single-GPU Edge backup.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
USER_SYSTEMD_DIR="${HOME}/.config/systemd/user"
SERVICE_NAME="${QWEN_TTS_SERVICE_NAME:-qwen3-tts-single}"
SERVICE_FILE="${USER_SYSTEMD_DIR}/${SERVICE_NAME}.service"

PYTHON_BIN="${QWEN_TTS_PYTHON:-$ROOT_DIR/.venv/bin/python}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0}"
MODEL_PATH="${QWEN_TTS_MODEL:-/home/admin/models/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
HOST="${QWEN_TTS_HOST:-0.0.0.0}"
PORT="${QWEN_TTS_PORT:-8091}"
DEVICE="${QWEN_TTS_DEVICE:-cuda:0}"
ATTN="${QWEN_TTS_ATTN:-sdpa}"
DTYPE="${QWEN_TTS_DTYPE:-float16}"
LANGUAGE="${QWEN_TTS_LANGUAGE:-Auto}"
SPEAKER="${QWEN_TTS_SPEAKER:-Serena}"
CHUNK_SIZE="${QWEN_TTS_CHUNK_SIZE:-8}"
MAX_NEW_TOKENS="${QWEN_TTS_MAX_NEW_TOKENS:-512}"
PROSODY_ENABLED="${QWEN_TTS_PROSODY_OPTIMIZER_ENABLED:-1}"
PROSODY_BASE_URL="${QWEN_TTS_PROSODY_BASE_URL:-http://agx.taild500c8.ts.net:11434/v1}"
PROSODY_MODEL="${QWEN_TTS_PROSODY_MODEL:-caps-voice-edit-qwen3-4b:latest}"
PROSODY_TIMEOUT_S="${QWEN_TTS_PROSODY_TIMEOUT_S:-2.5}"
PROSODY_MAX_TOKENS="${QWEN_TTS_PROSODY_MAX_TOKENS:-160}"
PROSODY_MAX_INPUT_CHARS="${QWEN_TTS_PROSODY_MAX_INPUT_CHARS:-1200}"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: Python not found or not executable: $PYTHON_BIN" >&2
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: model directory not found: $MODEL_PATH" >&2
    exit 1
fi

mkdir -p "$USER_SYSTEMD_DIR"

cat >"$SERVICE_FILE" <<EOF
[Unit]
Description=Faster Qwen3-TTS single-GPU backup service
After=network-online.target

[Service]
Type=simple
WorkingDirectory=${ROOT_DIR}
Environment=CUDA_VISIBLE_DEVICES=${GPU_LIST}
Environment=QWEN_TTS_PROSODY_OPTIMIZER_ENABLED=${PROSODY_ENABLED}
Environment=QWEN_TTS_PROSODY_BASE_URL=${PROSODY_BASE_URL}
Environment=QWEN_TTS_PROSODY_MODEL=${PROSODY_MODEL}
Environment=QWEN_TTS_PROSODY_TIMEOUT_S=${PROSODY_TIMEOUT_S}
Environment=QWEN_TTS_PROSODY_MAX_TOKENS=${PROSODY_MAX_TOKENS}
Environment=QWEN_TTS_PROSODY_MAX_INPUT_CHARS=${PROSODY_MAX_INPUT_CHARS}
ExecStart=${PYTHON_BIN} ${ROOT_DIR}/examples/single_gpu_custom_voice_server.py --model ${MODEL_PATH} --host ${HOST} --port ${PORT} --device ${DEVICE} --attn ${ATTN} --dtype ${DTYPE} --language ${LANGUAGE} --speaker ${SPEAKER} --chunk-size ${CHUNK_SIZE} --max-new-tokens ${MAX_NEW_TOKENS}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now "${SERVICE_NAME}.service"

echo "Installed and started: ${SERVICE_NAME}.service"
systemctl --user status "${SERVICE_NAME}.service" --no-pager -l | sed -n '1,80p'
