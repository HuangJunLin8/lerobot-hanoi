#!/bin/bash

TASK_NAME="$1"
HF_USER="rical"
SESSION="task"                      # 固定 session 名
WINDOW="train_${TASK_NAME}"         # 动态 window 名称

WAIT_INTERVAL=300
MAX_WAIT_HOURS=12
START_TIME=$(date +%s)

log() {
  echo "[$(date +%H:%M)] $1"
}

is_gpu_idle() {
  local threshold=20
  local busy_count=0
  for util in $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits); do
    if [ "$util" -ge "$threshold" ]; then
      busy_count=$((busy_count + 1))
    fi
  done
  if [ "$busy_count" -eq 0 ]; then
    return 0
  else
    return 1
  fi
}

log "等待 GPU 利用率低于 20%：$TASK_NAME"
while true; do
  if is_gpu_idle; then
    log "GPU 空闲，启动任务：$TASK_NAME"
    break
  fi
  NOW=$(date +%s)
  ELAPSED=$(( (NOW - START_TIME) / 60 ))
  if [ "$ELAPSED" -gt $((MAX_WAIT_HOURS * 60)) ]; then
    log "等待 GPU 超过 ${MAX_WAIT_HOURS} 小时，放弃任务：$TASK_NAME"
    exit 1
  fi
  log "GPU 忙碌中，等待 ${WAIT_INTERVAL}s 后重试... 已等待 ${ELAPSED} 分钟"
  sleep $WAIT_INTERVAL
done

# 在已存在的 session 中新建 window
tmux new-window -t ${SESSION} -n ${WINDOW}

tmux send-keys -t ${SESSION}:${WINDOW} "cd /home/data/lerobot/rical/lerobot-hanoi/" C-m
tmux send-keys -t ${SESSION}:${WINDOW} "conda activate lerobot" C-m

# 登陆 wandb 
tmux send-keys -t ${SESSION}:${WINDOW} "wandb login" C-m

tmux send-keys -t ${SESSION}:${WINDOW} "export TASK_NAME=${TASK_NAME}" C-m
tmux send-keys -t ${SESSION}:${WINDOW} "export HF_USER=${HF_USER}" C-m

# 删除之前训练的模型
tmux send-keys -t ${SESSION}:${WINDOW} "rm -rf outputs/train/${TASK_NAME}" C-m

# 开始训练
tmux send-keys -t ${SESSION}:${WINDOW} "python lerobot/scripts/train.py \
  --dataset.repo_id=\${HF_USER}/\${TASK_NAME} \
  --policy.type=act \
  --output_dir=outputs/train/\${TASK_NAME} \
  --job_name=\${TASK_NAME} \
  --policy.device=cuda \
  --wandb.enable=true" C-m
