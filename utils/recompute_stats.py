import argparse
import json
import os
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import EPISODES_STATS_PATH, INFO_PATH, EPISODES_PATH, load_stats, write_info
from lerobot.common.datasets.v21.convert_stats import convert_stats, check_aggregate_stats


def remove_old_stats(dataset):
    """删除旧 episodes_stats.jsonl 文件"""
    stats_path = dataset.root / EPISODES_STATS_PATH
    if stats_path.exists():
        stats_path.unlink()
        print("🧹 已删除旧 episodes_stats.jsonl")


def generate_new_stats(dataset, num_workers):
    """重新生成 episodes_stats.jsonl 文件"""
    convert_stats(dataset, num_workers=num_workers)
    print("✅ 新统计信息已生成：episodes_stats.jsonl")


def get_total_frames(episodes_path):
    """计算所有 episodes 的总帧数"""
    total_frames = 0
    if episodes_path.exists():
        with open(episodes_path, "r") as f:
            for line in f:
                episode = json.loads(line)
                total_frames += episode.get("length", 0)
    return total_frames


def get_total_episodes(episodes_path):
    """计算 episodes 数量"""
    if episodes_path.exists():
        with open(episodes_path, "r") as f:
            return sum(1 for _ in f)
    return 0


def get_total_videos(videos_path):
    """统计 videos 目录下所有视频数量"""
    return sum(len(files) for _, _, files in os.walk(videos_path) if files)


def update_info_json(basePath, total_frames, total_episodes, total_videos):
    """更新 info.json 文件"""
    info_path = basePath / INFO_PATH
    if info_path.exists():
        with open(info_path, "r") as f:
            info_data = json.load(f)
    else:
        info_data = {}

    info_data.update({
        "total_frames": total_frames,
        "total_episodes": total_episodes,
        "total_videos": total_videos,
    })

    write_info(info_data, basePath)
    print(f"📝 已更新 info.json: total_frames={total_frames}, total_episodes={total_episodes}, total_videos={total_videos}")


def recompute_stats_v21(repo_id: str, num_workers: int = 8, video_backend="pyav"):
    # 计算 meta 信息， 更新 info.json
    basePath = Path("~/.cache/huggingface/lerobot/" + repo_id ).expanduser()  # 要展开`~`
    episodes_path = basePath / EPISODES_PATH
    videos_path = basePath / "videos"
    episodes_stats_path = basePath /EPISODES_STATS_PATH

 
    total_frames = get_total_frames(episodes_path)
    total_episodes = get_total_episodes(episodes_path)
    total_videos = get_total_videos(videos_path)

    update_info_json(basePath, total_frames, total_episodes, total_videos)

    # 重新计算 episodes_stats.jsonl (info.json 得是正确的先)
    
    # 如果 episodes_stats.jsonl 文件不存在，就创建一个空文件(否则无法加载数据集)
    if not episodes_stats_path.exists():
        episodes_stats_path.parent.mkdir(parents=True, exist_ok=True)  
        episodes_stats_path.touch()

    dataset = LeRobotDataset(
        repo_id=repo_id,
        video_backend= video_backend,
        force_cache_sync=False,
    )
    print(f"📦 数据集加载成功：{repo_id}")

    remove_old_stats(dataset)
    generate_new_stats(dataset, num_workers)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="重新计算 LeRobot v2.1 数据集的统计信息和 meta 信息")

    parser.add_argument("--repo_id", type=str, required=True, help="本地数据集路径")
    parser.add_argument("--num-workers", type=int, default=4, help="并行计算的线程数")
    parser.add_argument("--video-backend", type=str, default="pyav", help="视频处理后端（pyav 或 ffmpeg）")

    args = parser.parse_args()

    recompute_stats_v21(
        repo_id=args.repo_id,
        num_workers=args.num_workers,
        video_backend=args.video_backend,
    )


"""
例子：
    python utils/recompute_stats.py --repo_id ${HF_USER}/${TASK_NAME}
"""