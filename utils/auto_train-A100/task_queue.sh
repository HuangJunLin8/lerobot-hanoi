#!/bin/bash
TASK_FILE="/mnt/data/lerobot/bin/tasks.txt"
SCRIPT_PATH="/mnt/data/lerobot/bin/run_training.sh"
SLEEP_INTERVAL=300  # 每轮检测间隔（秒）
WAIT_CONFIRM=120    # GPU 空闲后确认等待时间（秒）
GPU_IDLE_THRESHOLD=20  # GPU 利用率小于这个值才认为是空闲

log() {
  echo "[$(date +%H:%M)] $1"
}

check_gpu_idle() {

  # 检查是否有活跃的GPU进程
  local processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  if [ "$processes" -gt 0 ]; then
    return 1  # 有进程运行，视为繁忙
  fi

  return 0  # GPU空闲
}

while read -r TASK_NAME || [ -n "$TASK_NAME" ]; do
  log "等待执行任务: $TASK_NAME"

  while true; do
    if check_gpu_idle; then
      log "第一次检测到 GPU 空闲，等待 $WAIT_CONFIRM 秒再确认..."
      sleep $WAIT_CONFIRM
      if check_gpu_idle; then
        log "GPU 确认空闲，开始任务: $TASK_NAME"
        bash "$SCRIPT_PATH" "$TASK_NAME"
        break
      else
        log "二次检查发现 GPU 忙碌，任务推迟"
      fi
    else
      log "GPU 忙碌，等待 $SLEEP_INTERVAL 秒后重试..."
    fi
    sleep $SLEEP_INTERVAL
  done

done < "$TASK_FILE"
