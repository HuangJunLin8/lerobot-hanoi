import os
import shutil
import json
from collections import defaultdict
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
from recompute_stats import recompute_stats_v21
from lerobot.common.datasets.utils import TASKS_PATH, EPISODES_PATH, EPISODES_STATS_PATH


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


def merge_datasets(dataset_path1, dataset_path2, merged_dir):
    """ 
        合并两个数据集
        merged_dir: 合并后的数据集保存路径
        dataset_path1: 数据集1路径
        dataset_path2: 数据集2路径
    """
    dataset_path1 = Path(dataset_path1)
    dataset_path2 = Path(dataset_path2)
    merged_dir = Path(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)

    file_counters = defaultdict(int)
    combined_episodes = []
    combined_tasks = []

    for dataset_dir in [dataset_path1, dataset_path2]:
        # 处理 meta/episodes.jsonl
        meta_path = dataset_dir / 'meta' / 'episodes.jsonl'
        episode_mapping = {}
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                for line in f:
                    episode_data = json.loads(line.strip())
                    old_index = episode_data['episode_index']
                    file_counters['episode'] += 1
                    new_index = file_counters['episode'] - 1
                    episode_data['episode_index'] = new_index
                    episode_mapping[old_index] = new_index
                    combined_episodes.append(episode_data)

        # 处理 meta/tasks.jsonl
        task_path = dataset_dir / 'meta' / 'tasks.jsonl'
        if task_path.exists():
            with open(task_path, 'r') as f:
                for line in f:
                    task_data = json.loads(line.strip())
                    task_data['task_index'] = file_counters['task']  # 递增编号
                    file_counters['task'] += 1
                    combined_tasks.append(task_data)

        # 拷贝其他数据文件
        for root, _, files in os.walk(dataset_dir):
            relative_path = Path(root).relative_to(dataset_dir)
            for file in files:
                if file in ['episodes.jsonl', 'tasks.jsonl']:
                    continue  # meta 目录已独立处理
                src_file = Path(root) / file
                dest_dir = merged_dir / relative_path
                dest_dir.mkdir(parents=True, exist_ok=True)

                if "episode" in file:
                    file_stem, file_suffix = file.split(".")[0], f".{file.split('.')[-1]}"
                    if "episode_" in file_stem:
                        try:
                            old_index = int(file_stem.split("_")[1])
                            new_index = episode_mapping.get(old_index, old_index)
                            new_file_name = f"episode_{str(new_index).zfill(6)}{file_suffix}"
                        except ValueError:
                            new_file_name = file  # 比如 stats.parquet，不动
                    else:
                        new_file_name = file
                    dest_file = dest_dir / new_file_name
                    shutil.copy(src_file, dest_file)
                    if file_suffix == ".parquet" and "episode_" in file_stem:
                        modify_episode_index(dest_file, new_index)
                else:
                    dest_file = dest_dir / file
                    shutil.copy(src_file, dest_file)

    # 写入合并后的 meta/episodes.jsonl
    merged_meta_dir = merged_dir / 'meta'
    merged_meta_dir.mkdir(exist_ok=True)
    merged_episodes_path = merged_meta_dir / 'episodes.jsonl'
    with open(merged_episodes_path, 'w') as f:
        for episode in combined_episodes:
            f.write(json.dumps(episode) + '\n')

    # 写入合并后的 meta/tasks.jsonl
    merged_tasks_path = merged_meta_dir / 'tasks.jsonl'
    with open(merged_tasks_path, 'w') as f:
        for task in combined_tasks:
            f.write(json.dumps(task) + '\n')

    # 重新计算统计信息
    repoid = get_repo_id(merged_dir)
    recompute_stats_v21(repoid)

    print("✅ 两个数据集合并完成！")

if __name__ == "__main__":
    # 删除第 38 个 episode
    # base_path = "/home/rical/.cache/huggingface/lerobot/rical/A1234-B-C_mvA2B"
    # delete_episode(base_path, 19)

    # 重新计算数据集信息
    # repoid = get_repo_id(base_path)
    # recompute_stats_v21(repoid)

    # 删除数据集 [50, 54)
    # base_path = "/home/rical/.cache/huggingface/lerobot/rical/test"
    # delete_episode_range(base_path, 50, 54)
    
    # repoid = get_repo_id(base_path)
    # recompute_stats_v21(repoid)

    # 修改task名称
    # base_path = "/home/rical/.cache/huggingface/lerobot/rical/A1234-B-C_mvA2B"
    # modify_episode_tasks(base_path)

    # 合并两个数据集
    dataset_path1 = "/home/rical/.cache/huggingface/lerobot/rical/A12-B-C34_mvA2B"
    dataset_path2 = "/home/rical/.cache/huggingface/lerobot/rical/A1234-B-C_mvA2B"
    output = "/home/rical/.cache/huggingface/lerobot/rical/mvA2B"
    merge_datasets(dataset_path1, dataset_path2, output)