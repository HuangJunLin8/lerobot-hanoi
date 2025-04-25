#!/bin/bash

TASK_FILE="/home/data/lerobot/bin/tasks.txt"
SCRIPT_PATH="/home/data/lerobot/bin/run_training.sh"
START_TIME=$(date +%s)
INTERVAL=$((2 * 60 * 60 + 30 * 60))  # 2.5 小时（秒）

i=0
while read -r TASK_NAME; do
  SCHEDULE_TIME=$((START_TIME + i * INTERVAL))
  TIME_STR=$(date -d "@$SCHEDULE_TIME" "+%H:%M %Y-%m-%d")

  echo "Scheduling $TASK_NAME at $TIME_STR"
  echo "$SCRIPT_PATH $TASK_NAME" | at -M -t "$(date -d "@$SCHEDULE_TIME" +"%Y%m%d%H%M")"

  ((i++))
done < "$TASK_FILE"

