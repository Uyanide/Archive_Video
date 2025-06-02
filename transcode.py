# transcode.py
# 将一个视频文件按 ARGS 中定义的所有编码方式和参数进行转码输出多个文件

import os
import sys
from pathlib import Path
import queue
import threading
import prettytable

import console_log as log
from proc import TranscodeTask
from args import ARGS


OUTPUT_DIR = "archived"  # by default
OUTPUT_RESULT_NAME = "transcode_results.csv"
ENABLE_VMAF = True


# 3840x2160
#
# "-multipass", "fullres",
# "-rc", "vbr_hq", # or "vbr" for av1
# "-cq", "20",
# "-b:v", "20M",
# "-maxrate", "30M",
# "-bufsize", "60M",
# "-preset", "p7",
# "-profile:v", "main10", # or "high" for h264
# "-pix_fmt", "p010le", # or "yuv420p" for h264
# +----------------------+------------+----------+----------+----------+----------+--------+----------+
# |        文件名        |   编码器   | 原始大小 | 转码大小 | 原始码率 | 转码码率 | 压缩率 | VMAF平均 |
# +----------------------+------------+----------+----------+----------+----------+--------+----------+
# | park_joy_2160p50.y4m | av1_nvenc  |  5.8GB   |  37.4MB  | 4.6gbps  | 29.9mbps |  0.6%  |  84.46   |
# | park_joy_2160p50.y4m | hevc_nvenc |  5.8GB   |  36.2MB  | 4.6gbps  | 29.0mbps |  0.6%  |  80.81   |
# | park_joy_2160p50.y4m | h264_nvenc |  5.8GB   |  35.3MB  | 4.6gbps  | 28.2mbps |  0.6%  |  77.53   |
# +----------------------+------------+----------+----------+----------+----------+--------+----------+

# 2400x1440
#
# "-multipass", "fullres",
# "-rc", "vbr_hq", # or "vbr" for av1
# "-cq", "{}", # 32 for av1, 28 for hevc, 23 for h264
# "-b:v", "2M",
# "-maxrate", "5M",
# "-bufsize", "20M",
# "-preset", "p7",
# "-profile:v", "main10", # or "high" for h264
# "-pix_fmt", "p010le", # or "yuv420p" for h264
# +--------------------------------------+------------+----------+----------+----------+----------+--------+----------+
# |                文件名                |   编码器   | 原始大小 | 转码大小 | 原始码率 | 转码码率 | 压缩率 | VMAF平均 |
# +--------------------------------------+------------+----------+----------+----------+----------+--------+----------+
# | MuMu模拟器12 2025-05-28 04-27-04.mp4 | av1_nvenc  |  61.0MB  |  11.1MB  | 14.4mbps | 2.6mbps  | 18.2%  |  95.09   |
# | MuMu模拟器12 2025-05-28 04-27-04.mp4 | hevc_nvenc |  61.0MB  |  10.2MB  | 14.4mbps | 2.4mbps  | 16.7%  |  94.11   |
# | MuMu模拟器12 2025-05-28 04-27-04.mp4 | h264_nvenc |  61.0MB  |  14.9MB  | 14.4mbps | 3.5mbps  | 24.4%  |  93.94   |
# +--------------------------------------+------------+----------+----------+----------+----------+--------+----------+


def transcode_worker(input_file: Path, output_path: Path, tasks: queue.Queue[TranscodeTask | None], transcode_finished: threading.Event) -> None:
    for index, (encoder, encode_args) in enumerate(ARGS.items(), 1):
        output_file = output_path / f"{input_file.stem}_{encode_args.encoder}{encode_args.ext_name}"
        log.log_info(f"正在处理 ({index}/{len(ARGS)}): {input_file} -> {output_file}")
        task = TranscodeTask(input_file, output_file)
        if not task.transcode(encode_args):
            log.log_error(f"转码失败: {input_file.name}")
            tasks.put(task)
            continue
        task.get_compression_rate()
        task.get_duration()
        task.get_bitrate()
        res_status = task.to_status()
        if res_status:
            log.log_success(res_status)
        tasks.put(task)
    log.log_info(f"所有视频转码任务已提交，等待评估结果...")
    transcode_finished.set()
    tasks.put(None)


def evaluate_worker(tasks: queue.Queue[TranscodeTask], results: list[TranscodeTask | None], log_path: Path, enable_vmaf: bool, transcode_finished: threading.Event) -> None:
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(TranscodeTask.csv_header() + "\n")
        while True:
            task = tasks.get()
            if task is None:
                tasks.task_done()
                break
            if not task.failed and enable_vmaf:
                log.log_info(f"正在评估 VMAF: {task.input_file.name} (队列中剩余 {tasks.qsize()})")
                if task.get_vmaf_score(log_path.parent) is not None:
                    res_status = task.to_status()
                    if res_status:
                        log.log_success(res_status)

            log_file.write(task.to_csv() + "\n")
            log_file.flush()
            results.append(task)
            tasks.task_done()


def print_results(results: list[TranscodeTask]) -> None:
    """Print the results in a pretty table."""
    if not results:
        return
    table = prettytable.PrettyTable()
    table.field_names = TranscodeTask.rows()
    for task in results:
        table.add_row(task.to_row())
    print(table)


def main(input_file: str, output_dir: str = OUTPUT_DIR, enable_vmaf: bool = ENABLE_VMAF) -> None:
    os.environ["PYTHONUTF8"] = "1"

    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: list[TranscodeTask] = []
    tasks = queue.Queue()
    transcode_finished = threading.Event()

    evaluate_thread = threading.Thread(
        target=evaluate_worker,
        name="评估",
        args=(tasks, results, output_path / OUTPUT_RESULT_NAME, enable_vmaf, transcode_finished)
    )
    evaluate_thread.start()
    transcode_thread = threading.Thread(
        target=transcode_worker,
        name="转码",
        args=(input_path, output_path, tasks, transcode_finished)
    )
    transcode_thread.start()
    transcode_thread.join()
    evaluate_thread.join()

    log.log_success(f"所有视频处理完成，结果已保存为 {output_path / OUTPUT_RESULT_NAME}")
    print_results(results)


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            if len(sys.argv) == 3:
                main(sys.argv[1], sys.argv[2])
            else:
                main(sys.argv[1])
        else:
            print("用法: python transcode.py <输入文件> [输出目录]")
    except KeyboardInterrupt:
        log.log_error("用户中断")
        sys.exit(130)
