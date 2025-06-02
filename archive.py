# archive.py
# 将所有输入文件一一对应转码输出至指定目录

import os
import sys
from pathlib import Path
import queue
import threading
import prettytable
from argparse import ArgumentParser

import console_log as log
from proc import TranscodeTask
from args import ARGS


OUTPUT_DIR = "archived"  # by default
OUTPUT_RESULT_NAME = "archive_results.csv"
DEFAULT_ENCODER = "hevc_nvenc"


def transcode_worker(inputs: list[TranscodeTask], tasks: queue.Queue[TranscodeTask | None], transcode_finished: threading.Event) -> None:
    for index, task in enumerate(inputs, 1):
        if not task.encoder:
            task.encoder = DEFAULT_ENCODER
        log.log_info(f"正在处理 ({index}/{len(inputs)}): {task.input_file} -> {task.output_file} ({task.encoder})")
        args = ARGS[task.encoder]
        if not args:
            log.log_error(f"未找到编码器参数: {task.encoder}")
            task.mark_failed()
            tasks.put(task)
            continue
        if not task.transcode(args):
            log.log_error(f"转码失败: {task.input_file.name}")
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


def main(input_files: list[Path], output_dir: Path, enable_vmaf: bool, encoder: str, overwrite: bool) -> None:
    os.environ["PYTHONUTF8"] = "1"

    output_dir.mkdir(parents=True, exist_ok=True)

    inputs: list[TranscodeTask] = []
    results: list[TranscodeTask] = []
    tasks = queue.Queue[TranscodeTask | None]()

    for input_file in input_files:
        if not input_file.is_file():
            log.log_error(f"输入文件不存在: {input_file}")
            continue
        output_file = (output_dir / input_file.stem).with_suffix(ARGS[encoder].ext_name)
        if output_file.exists() and not output_file.is_file():
            log.log_error(f"输出路径错误: {output_file}, 跳过 {input_file}")
            continue
        if output_file.resolve() == input_file.resolve():
            output_file = (output_dir / (input_file.stem + "_archived")).with_suffix(ARGS[encoder].ext_name)
            log.log_warning(f"输出文件与输入文件相同: {input_file}, 将输出为 {output_file}")
        if output_file.exists():
            if not overwrite:
                log.log_warning(f"输出文件已存在: {output_file}")
                replace = input(f"是否替换？(y/n): ").strip().lower()
                if replace != 'y':
                    log.log_info(f"跳过文件: {input_file}")
                    continue
        task = TranscodeTask(input_file, output_file)
        task.encoder = encoder
        inputs.append(task)

    transcode_finished = threading.Event()
    transcode_thread = threading.Thread(
        target=transcode_worker,
        name="转码",
        args=(inputs, tasks, transcode_finished)
    )
    evaluate_thread = threading.Thread(
        target=evaluate_worker,
        name="评估",
        args=(tasks, results, output_dir / OUTPUT_RESULT_NAME, enable_vmaf, transcode_finished)
    )
    evaluate_thread.start()
    transcode_thread.start()
    transcode_thread.join()
    evaluate_thread.join()

    log.log_success(f"所有视频处理完成，结果已保存为 {output_dir / OUTPUT_RESULT_NAME}")
    print_results(results)


def parse_patterns(patterns: list[str]) -> list[Path]:
    """解析输入的通配符模式，返回匹配的文件列表"""
    files = []
    for pattern in patterns:
        files.extend(Path(p).resolve() for p in Path().glob(pattern))
    return sorted(set(files), key=lambda p: p.name)


if __name__ == "__main__":
    try:
        parser = ArgumentParser(description="视频归档")
        parser.add_argument(
            "input_files",
            nargs="+",
            type=str,
            help="输入视频文件路径，支持通配符"
        )
        parser.add_argument(
            "-o", "--output_dir",
            type=Path,
            default=Path(OUTPUT_DIR),
            help=f"输出目录，默认为 {OUTPUT_DIR}"
        )
        parser.add_argument(
            "-v", "--enable_vmaf",
            action="store_true",
            default=False,
            help="启用 VMAF 评估"
        )
        parser.add_argument(
            "-e", "--encoder",
            type=str,
            default=DEFAULT_ENCODER,
            choices=list(ARGS.keys()),
            help=f"指定编码器，默认为 {DEFAULT_ENCODER}"
        )
        parser.add_argument(
            "-y",
            action="store_true",
            default=False,
            help="自动确认覆盖已存在的输出文件，无需手动输入"
        )
        args = parser.parse_args()
        inputs = parse_patterns(args.input_files)
        if not inputs:
            log.log_error("未找到匹配的输入文件")
            sys.exit(1)
        log.log_info(f"找到 {len(inputs)} 个输入文件，开始处理...")
        main(inputs, args.output_dir, args.enable_vmaf, args.encoder, args.y)
    except KeyboardInterrupt:
        log.log_error("用户中断")
        sys.exit(130)
