#!/bin/bash

TASK_NAME="$1"
HF_USER="rical"
SESSION="task"                      # 固定 session 名称
WINDOW="train_${TASK_NAME}"         # 动态 window 名称

# 在 task session 里新开一个 window，名字叫 train_${TASK_NAME}
# 不能嵌套开
tmux new-window -t ${SESSION} -n ${WINDOW}

# 发送命令到新开的 window
tmux send-keys -t ${SESSION}:${WINDOW} "docker exec -it lerobot /bin/zsh" C-m
tmux send-keys -t ${SESSION}:${WINDOW} "cd /root/lerobot-hanoi" C-m
tmux send-keys -t ${SESSION}:${WINDOW} "conda activate lerobot-env" C-m
tmux send-keys -t ${SESSION}:${WINDOW} "pip install -e ." C-m

# 登陆 wandb 
tmux send-keys -t ${SESSION}:${WINDOW} "wandb login" C-m

# 设置环境变量
tmux send-keys -t ${SESSION}:${WINDOW} "export TASK_NAME=${TASK_NAME}" C-m
tmux send-keys -t ${SESSION}:${WINDOW} "export HF_USER=${HF_USER}" C-m

# 删除之前的模型目录
tmux send-keys -t ${SESSION}:${WINDOW} "rm -rf outputs/train/${TASK_NAME}" C-m

# 开始训练
tmux send-keys -t ${SESSION}:${WINDOW} "python lerobot/scripts/train.py \
  --dataset.repo_id=\${HF_USER}/\${TASK_NAME} \
  --policy.type=act \
  --output_dir=outputs/train/\${TASK_NAME} \
  --job_name=\${TASK_NAME} \
  --policy.device=cuda \
  --wandb.enable=true" C-m
