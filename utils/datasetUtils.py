import os
import shutil
import json
from collections import defaultdict
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
from recompute_stats import recompute_stats_v21
from lerobot.common.datasets.utils import EPISODES_PATH


def get_repo_id(path):
    """从路径中获得 repoid """
    path = Path(path)
    parts = path.parts
    if "lerobot" in parts:
        idx = parts.index("lerobot")
        repo_id = "/".join(parts[idx+1:idx+3])  # "rical/mvA2B"
        return repo_id
    else:
        raise ValueError("路径中找不到 'lerobot' 目录")


def modify_episode_index(file_path, episode_index):
    """修正元数据 episode_XXXXXX.parquet"""
    try:
        table = pq.read_table(file_path)
        df = table.to_pandas()
        if "episode_index" not in df.columns:
            raise ValueError("DataFrame 中没有 'episode_index' 列。")
        df["episode_index"] = episode_index
        table = pa.Table.from_pandas(df)
        pq.write_table(table, file_path)
    except Exception as e:
        print(f"操作失败: {e}")


def modify_episode_tasks(base_path, task_name=None):
    """
    修改 episodes.jsonl 中所有 episode 的 tasks 字段
    以及 tasks.jsonl 的 task 字段
    仅适合数据集中仅有一个task的情况

    参数:
        base_path (str | Path): 数据集根目录路径。
        task_name (str): 替换的新的 task 名称。
    """
    base_path = Path(base_path)

    # task name 默认为数据集的名字
    if task_name is None:
        task_name = base_path.name

    # 修改 episodes.jsonl
    episodes_path = base_path / EPISODES_PATH
    with open(episodes_path, 'r', encoding='utf-8') as infile:
        episodes = [json.loads(line) for line in infile]

    for episode in episodes:
        episode["tasks"] = [task_name]

    with open(episodes_path, 'w', encoding='utf-8') as outfile:
        for episode in episodes:
            json.dump(episode, outfile)
            outfile.write('\n')

    # 修改 tasks.jsonl
    task_path = base_path / TASKS_PATH
    if task_path.exists():
        with open(task_path, 'r', encoding='utf-8') as f:
            task_data = json.loads(f.readline())
        task_data["task"] = task_name
        with open(task_path, 'w', encoding='utf-8') as f:
            json.dump(task_data, f)
            f.write('\n')

    print(f"✅ 修改完成")


def delete_episode(base_path, episode_index):
    """删除特定数据集

    Args:
        base_path (str): 数据集路径
        episode_index (int): 删除的数据集下标
    """
    base_path = Path(base_path)
    data_path = base_path / 'data'
    videos_path = base_path / 'videos'
    episodes_file = base_path / EPISODES_PATH 

    for root, _, files in os.walk(data_path):
        files.sort()
        for file in files:
            if file.startswith("episode_"):
                episode_num = int(file.split("_")[1].split(".")[0])
                src_file = Path(root) / file
                if episode_num == episode_index:
                    os.remove(src_file)
                elif episode_num > episode_index:
                    new_num = episode_num - 1
                    new_file_name = f"episode_{str(new_num).zfill(6)}.parquet"
                    dest_file = Path(root) / new_file_name
                    os.rename(src_file, dest_file)
                    modify_episode_index(dest_file, new_num)

    for root, _, files in os.walk(videos_path):
        files.sort()
        for file in files:
            if file.startswith("episode_"):
                episode_num = int(file.split("_")[1].split(".")[0])
                src_file = Path(root) / file
                if episode_num == episode_index:
                    os.remove(src_file)
                elif episode_num > episode_index:
                    new_num = episode_num - 1
                    new_file_name = f"episode_{str(new_num).zfill(6)}{Path(file).suffix}"
                    dest_file = Path(root) / new_file_name
                    os.rename(src_file, dest_file)

    updated_episodes = []
    if episodes_file.exists():
        with open(episodes_file, 'r') as f:
            episodes = [json.loads(line) for line in f]
        for episode in episodes:
            if episode['episode_index'] != episode_index:
                if episode['episode_index'] > episode_index:
                    episode['episode_index'] -= 1
                updated_episodes.append(episode)
        with open(episodes_file, 'w') as f:
            for episode in updated_episodes:
                f.write(json.dumps(episode) + '\n')

    print(f"已从数据集 {base_path.name} 删除 episode{episode_index}")


def delete_episode_range(base_path, start_index, end_index):
    """
    删除指定区间 [start_index, end_index) 的 episode 数据，

    参数:
        base_path (str): 数据集的基本路径。
        start_index (int): 删除的起始 episode_index（包含）。
        end_index (int): 删除的结束 episode_index（不包含）。
    """

    base_path = Path(base_path)
    
    # 先按从大到小顺序删除，防止 episode_index 变化导致问题
    for episode_index in range(end_index - 1, start_index - 1, -1):
        delete_episode(base_path, episode_index)
    
    print(f"数据集{base_path.name} 中 [{start_index}, {end_index}) 的 episode 删除完成！")




if __name__ == "__main__":
    # 删除第 38 个 episode
    # base_path = "/home/rical/.cache/huggingface/lerobot/rical/test"
    # delete_episode(base_path, 38)

    # 重新计算数据集信息
    # repoid = get_repo_id(base_path)
    # recompute_stats_v21(repoid)

    # 删除数据集 [50, 54)
    # base_path = "/home/rical/.cache/huggingface/lerobot/rical/test"
    # delete_episode_range(base_path, 50, 54)
    
    # repoid = get_repo_id(base_path)
    # recompute_stats_v21(repoid)

    # 修改task名称
    base_path = "/home/rical/.cache/huggingface/lerobot/rical/A1234-B-C_mvA2B"
    modify_episode_tasks(base_path)