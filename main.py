import os
import sys
from pathlib import Path
import json
import queue
import threading
import prettytable
import hashlib
import ffmpeg

FFMPEG_PATH = R"D:\0-temp\3FUI\ffmpeg.exe"
FFPROBE_PATH = R"D:\0-temp\3FUI\ffprobe.exe"

OUTPUT_DIR = "archived"  # default
OUTPUT_RESULT_NAME = "transcode_results.csv"

ENABLE_VMAF = True
VMAF_N_THREADS = 4
VMAF_SAMPLE_FRAMES = 1000

GLOBAL_ARS = [
    '-hide_banner'
]


class EncodeArgs:
    def __init__(self, encoder: str, ext_name: str, params: list[str]) -> None:
        self.encoder: str = encoder
        self.ext_name: str = ext_name
        self.codec_params: list[str] = params


ARGS: list[EncodeArgs] = [
    EncodeArgs(
        "av1_nvenc", ".mkv",
        [
            "-c:v", "av1_nvenc",
            "-multipass", "fullres",
            "-rc", "vbr",
            "-cq", "32",
            "-b:v", "2M",
            "-maxrate", "5M",
            "-bufsize", "20M",
            "-preset", "p7",
            "-profile:v", "main10",
            "-pix_fmt", "p010le",
        ]),
    EncodeArgs(
        "hevc_nvenc", ".mkv",
        [
            "-c:v", "hevc_nvenc",
            "-multipass", "fullres",
            "-rc", "vbr_hq",
            "-cq", "28",
            "-b:v", "2M",
            "-maxrate", "5M",
            "-bufsize", "20M",
            "-preset", "p7",
            "-profile:v", "main10",
            "-pix_fmt", "p010le",
        ]),
    EncodeArgs(
        "h264_nvenc", ".mkv",
        [
            "-c:v", "h264_nvenc",
            "-multipass", "fullres",
            "-rc", "vbr_hq",
            "-cq", "23",
            "-b:v", "2M",
            "-maxrate", "5M",
            "-bufsize", "20M",
            "-preset", "p7",
            "-profile:v", "high",
            "-pix_fmt", "yuv420p",
        ]),
]


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

_log_lock = threading.Lock()


class Log:
    @staticmethod
    def _get_thread_prefix() -> str:
        thread_name = threading.current_thread().name
        if thread_name == "MainThread" and threading.current_thread() is not threading.main_thread():
            return f"[Thread-{threading.get_ident()}] "
        return f"[{thread_name}] "

    @staticmethod
    def log_error(message: str) -> None:
        """Log an error message to stderr."""
        with _log_lock:
            prefix = Log._get_thread_prefix()
            print(f"\033[31m{prefix}🟥 [ERROR]\033[0m ", end="", file=sys.stderr)
            print(message, file=sys.stderr)
            print("", file=sys.stderr)

    @staticmethod
    def log_warning(message: str) -> None:
        """Log a warning message to stderr."""
        with _log_lock:
            prefix = Log._get_thread_prefix()
            print(f"\033[33m{prefix}🟨 [WARN]\033[0m ", end="", file=sys.stderr)
            print(message, file=sys.stderr)
            print("", file=sys.stderr)

    @staticmethod
    def log_success(message: str) -> None:
        """Log a success message to stdout."""
        with _log_lock:
            prefix = Log._get_thread_prefix()
            print(f"\033[32m{prefix}🟩 [SUCCESS]\033[0m ", end="")
            print(message)
            print("")

    @staticmethod
    def log_info(message: str) -> None:
        """Log an informational message to stdout."""
        with _log_lock:
            prefix = Log._get_thread_prefix()
            print(f"\033[34m{prefix}🟦 [INFO]\033[0m ", end="")
            print(message)
            print("")

    @staticmethod
    def log_debug(message: str) -> None:
        """Log a debug message to stdout."""
        with _log_lock:
            prefix = Log._get_thread_prefix()
            print(f"\033[37m{prefix}⬜ [DEBUG]\033[0m ", end="", file=sys.stderr)
            print(message, file=sys.stderr)
            print("", file=sys.stderr)


def _format_size(bytes_size, lower_case: bool = False):
    if bytes_size == 0:
        return "0B"

    units = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    index = 0
    while bytes_size >= 1024 and index < len(units) - 1:
        bytes_size /= 1024
        index += 1
    result = f"{bytes_size:.1f}{units[index]}"
    return result.lower() if lower_case else result


def _delete_file(file_path: Path) -> None:
    """Delete a file if it exists. Handles exceptions silently."""
    if file_path.exists():
        try:
            file_path.unlink()
        except Exception as e:
            pass


def _params_to_kwargs(params_list: list[str]) -> dict[str, str]:
    """Converts a list of ffmpeg CLI parameters to a dictionary for ffmpeg-python."""
    # Assumes parameters are in "-key", "value" pairs.
    return {params_list[i].lstrip('-'): params_list[i+1] for i in range(0, len(params_list), 2)}


class TranscodeTask:

    def init_values(self, input_file: Path, output_file: Path, failed: bool = False) -> None:
        self.failed: bool = failed
        self.isDone: bool = failed  # failure is considered done
        self.encoder: str | None = None
        self.input_file: Path = input_file
        self.output_file: Path = output_file
        self.orig_size: int | None = None
        self.dist_size: int | None = None
        self.comp_rate: float | None = None
        self.vmaf_min: float | None = None
        self.vmaf_max: float | None = None
        self.vmaf_mean: float | None = None
        self.vmaf_harmonic_mean: float | None = None
        self.orig_duration: float | None = None
        self.dist_duration: float | None = None
        self.orig_bitrate: int | None = None
        self.dist_bitrate: int | None = None
        self.frames_count: int | None = None

    def __init__(self, input_file: Path, output_file: Path, failed: bool = False) -> None:
        self.init_values(input_file, output_file, failed)

    def mark_failed(self) -> None:
        """Mark as failed."""
        self.init_values(self.input_file, self.output_file, failed=True)

    def to_status(self) -> str:
        """Construct the results."""
        status = ""
        if self.encoder is not None:
            status += f"| {self.encoder} "
        if self.comp_rate is not None:
            status += f"| {self.comp_rate * 100:.1f}% 压缩率 | {_format_size(self.orig_size)} -> {_format_size(self.dist_size)} "
        if self.orig_bitrate is not None and self.dist_bitrate is not None:
            status += f"| {_format_size(self.orig_bitrate, lower_case=True)}ps -> {_format_size(self.dist_bitrate, lower_case=True)}ps "
        if self.vmaf_mean is not None:
            status += f"| VMAF (mean): {self.vmaf_mean:.2f} "
        if status:
            status += "|"
        return status

    @staticmethod
    def csv_header() -> str:
        """Return the CSV header for the results."""
        return "input_file,output_file,orig_size,dist_size,orig_bitrate,dist_bitrate,comp_rate,vmaf_min,vmaf_max,vmaf_mean,vmaf_harmonic_mean"

    def to_csv(self) -> str:
        """Convert the result to a CSV string."""
        orig_size = _format_size(self.orig_size) if self.orig_size is not None else "未知"
        comp_size = _format_size(self.dist_size) if self.dist_size is not None else "未知"
        orig_bitrate = f"{_format_size(self.orig_bitrate, lower_case=True)}ps" if self.orig_bitrate is not None else "未知"
        comp_bitrate = f"{_format_size(self.dist_bitrate, lower_case=True)}ps" if self.dist_bitrate is not None else "未知"
        comp_rate = f"{self.comp_rate * 100:.1f}%" if self.comp_rate is not None else "未知"
        vmaf_min = f"{self.vmaf_min:.2f}" if self.vmaf_min is not None else "未知"
        vmaf_max = f"{self.vmaf_max:.2f}" if self.vmaf_max is not None else "未知"
        vmaf_mean = f"{self.vmaf_mean:.2f}" if self.vmaf_mean is not None else "未知"
        vmaf_harmonic_mean = f"{self.vmaf_harmonic_mean:.2f}" if self.vmaf_harmonic_mean is not None else "未知"

        return f"{self.input_file},{self.output_file},{orig_size},{comp_size},{orig_bitrate},{comp_bitrate},{comp_rate},{vmaf_min},{vmaf_max},{vmaf_mean},{vmaf_harmonic_mean}"

    @staticmethod
    def rows() -> list[str]:
        """Return the header row for the pretty table."""
        return [
            "文件名", "编码器", "原始大小", "转码大小",
            "原始码率", "转码码率", "压缩率", "VMAF平均"
        ]

    def to_row(self) -> list[str]:
        """Convert the result to a row for the pretty table."""
        encoder = self.encoder if self.encoder else "未知"
        orig_size = _format_size(self.orig_size) if self.orig_size is not None else "未知"
        comp_size = _format_size(self.dist_size) if self.dist_size is not None else "未知"
        orig_bitrate = f"{_format_size(self.orig_bitrate, lower_case=True)}ps" if self.orig_bitrate is not None else "未知"
        comp_bitrate = f"{_format_size(self.dist_bitrate, lower_case=True)}ps" if self.dist_bitrate is not None else "未知"
        comp_rate = f"{self.comp_rate * 100:.1f}%" if self.comp_rate is not None else "未知"
        vmaf_mean = f"{self.vmaf_mean:.2f}" if self.vmaf_mean is not None else "未知"

        return [
            self.input_file.name, encoder, orig_size, comp_size,
            orig_bitrate, comp_bitrate, comp_rate, vmaf_mean
        ]

    def get_compression_rate(self) -> float | None:
        """Calculate the compression rate based on file sizes."""
        if not self.isDone:
            return None

        def get_file_size(path: Path) -> int | None:
            try:
                size = path.stat().st_size
                if size <= 0:
                    return None
                return size
            except FileNotFoundError:
                return None
            except Exception as e:
                return None
        if self.comp_rate is not None:
            return self.comp_rate
        if self.orig_size is None:
            self.orig_size = get_file_size(self.input_file)
        if self.dist_size is None:
            self.dist_size = get_file_size(self.output_file)
        if self.orig_size is not None and self.dist_size is not None:
            self.comp_rate = self.dist_size / self.orig_size
        return self.comp_rate

    def transcode(self, args: EncodeArgs) -> bool:
        """Transcode the video file using the specified encoding arguments."""
        if self.isDone:
            self.init_values(self.input_file, self.output_file, failed=False)

        ffmpeg_codec_options = _params_to_kwargs(args.codec_params)
        Log.log_debug(f"转码参数: {ffmpeg_codec_options}")

        try:
            stream = ffmpeg.input(str(self.input_file))
            stream = ffmpeg.output(stream, str(self.output_file), **ffmpeg_codec_options).global_args(*GLOBAL_ARS)

            Log.log_debug(f"执行命令 (转码): ffmpeg {' '.join(stream.get_args())}")
            stream.run(cmd=FFMPEG_PATH, quiet=True, overwrite_output=True, capture_stderr=True)

            self.isDone = True
            self.encoder = args.encoder
            return True
        except ffmpeg.Error as e:
            error_message = e.stderr.decode('utf-8', errors='replace') if e.stderr else str(e)
            Log.log_error(f"转码失败: {error_message}")
        except Exception as e:
            Log.log_error(f"转码时发生意外错误: {str(e)}")

        self.mark_failed()
        _delete_file(self.output_file)
        return False

    def get_vmaf_score(self, log_dir: str = OUTPUT_DIR) -> float | None:
        """Get the VMAF score from the result."""
        if not self.isDone or self.failed:
            return None
        if not self.frames_count:
            self.get_frames_count()
        if not self.frames_count:  # Ensure frames_count is available
            Log.log_warning(f"无法获取 {self.input_file.name} 的帧数，跳过 VMAF 计算。")
            return None

        if self.vmaf_mean is not None:  # Already calculated
            return self.vmaf_mean

        vmaf_log_path = Path(log_dir) / f"{hashlib.sha256(self.output_file.stem.encode('utf-8')).hexdigest()[:16]}_vmaf.json"
        try:
            Path(log_dir).mkdir(parents=True, exist_ok=True)

            n_subsample_val = max(self.frames_count // VMAF_SAMPLE_FRAMES, 1)

            distorted = ffmpeg.input(str(self.output_file))
            original = ffmpeg.input(str(self.input_file))

            vmaf_filter_args = {
                'log_fmt': 'json',
                'log_path': str(vmaf_log_path),
                'n_threads': VMAF_N_THREADS,
                'n_subsample': n_subsample_val
            }

            stream = ffmpeg.filter(
                [distorted, original],
                'libvmaf',
                **vmaf_filter_args
            )

            Log.log_debug(f"执行命令 (VMAF): ffmpeg {' '.join(ffmpeg.output(stream, '-', format='null').get_args())}")
            (ffmpeg.output(stream, '-', format='null')
             .global_args(*GLOBAL_ARS)
             .run(cmd=FFMPEG_PATH, quiet=True, capture_stderr=True))

            if not vmaf_log_path.exists():
                Log.log_error(f"VMAF 日志文件 {vmaf_log_path} 未创建。")
                return None

            with open(vmaf_log_path, "r", encoding="utf-8") as f:
                log_data = json.load(f)
                vmaf_dict = log_data.get("pooled_metrics", {}).get("vmaf", {})
                ret: dict[str, float | None] = {}
                for key in ["min", "max", "mean", "harmonic_mean"]:
                    value = vmaf_dict.get(key)
                    if value is None:
                        Log.log_warning(f"VMAF 日志缺少字段: {key}")
                        ret[key] = None
                    else:
                        try:
                            ret[key] = float(value)
                        except ValueError:
                            Log.log_warning(f"VMAF 字段 {key} 不是数字: {value}")
                            ret[key] = None
                self.vmaf_min = ret.get("min")
                self.vmaf_max = ret.get("max")
                self.vmaf_mean = ret.get("mean")
                self.vmaf_harmonic_mean = ret.get("harmonic_mean")
            return self.vmaf_mean

        except ffmpeg.Error as e:
            error_message = e.stderr.decode('utf-8', errors='replace') if e.stderr else str(e)
            Log.log_error(f"{self.input_file.name} VMAF 对比失败 - {error_message[:200]}")
            return None
        except json.JSONDecodeError as e:
            Log.log_error(f"解析 VMAF 日志失败 ({vmaf_log_path}): {str(e)}")
            return None
        except Exception as e:
            Log.log_error(f"VMAF 计算时发生意外错误: {str(e)}")
            return None
        finally:
            _delete_file(vmaf_log_path)

    def get_duration(self) -> tuple[float | None, float | None]:
        """Get the duration of the original and output video files."""
        if self.orig_duration is not None and self.dist_duration is not None:
            return self.orig_duration, self.dist_duration

        def _fetch_duration(path: Path) -> float | None:
            try:
                Log.log_debug(f"获取视频时长: {path.name}")
                probe = ffmpeg.probe(str(path), cmd=FFPROBE_PATH, show_entries='format=duration')
                duration_str = probe.get('format', {}).get('duration')
                if duration_str:
                    return float(duration_str)
                Log.log_warning(f"无法从 {path.name} 获取时长信息。")
            except ffmpeg.Error as e:
                error_message = e.stderr.decode('utf-8', errors='replace') if e.stderr else str(e)
                Log.log_error(f"获取视频时长失败 ({path.name}): {error_message}")
            except ValueError:
                Log.log_error(f"视频时长格式无效 ({path.name})")
            except Exception as e:
                Log.log_error(f"获取视频时长时发生意外错误 ({path.name}): {str(e)}")
            return None

        if self.orig_duration is None:
            self.orig_duration = _fetch_duration(self.input_file)

        if self.dist_duration is None and self.output_file.exists() and self.isDone and not self.failed:
            self.dist_duration = _fetch_duration(self.output_file)
        elif not self.output_file.exists() and self.isDone and not self.failed:
            Log.log_warning(f"输出文件 {self.output_file.name} 不存在，无法获取时长。")

        return self.orig_duration, self.dist_duration

    def get_bitrate(self) -> tuple[float | None, float | None]:
        """Get the bitrate of the original and output video files."""
        if self.orig_bitrate is not None and self.dist_bitrate is not None:
            return self.orig_bitrate, self.dist_bitrate
        if self.orig_size is None or self.dist_size is None:
            self.get_compression_rate()
        if self.orig_duration is None or self.dist_duration is None:
            self.get_duration()
        if self.orig_duration is not None and self.orig_size is not None:
            self.orig_bitrate = int((self.orig_size * 8) / self.orig_duration) if self.orig_duration > 0 else None
        if self.dist_duration is not None and self.dist_size is not None:
            self.dist_bitrate = int((self.dist_size * 8) / self.dist_duration) if self.dist_duration > 0 else None
        return self.orig_bitrate, self.dist_bitrate

    def get_frames_count(self) -> int | None:
        if self.frames_count is not None:
            return self.frames_count

        path_to_probe = self.input_file

        # Method 1: nb_frames
        try:
            Log.log_debug(f"获取帧数 (nb_frames): {path_to_probe.name}")
            probe_data = ffmpeg.probe(str(path_to_probe), cmd=FFPROBE_PATH,
                                      select_streams='v:0', show_entries='stream=nb_frames')
            if probe_data and 'streams' in probe_data and probe_data['streams']:
                nb_frames_str = str(probe_data['streams'][0].get('nb_frames', ''))
                if nb_frames_str.isdigit():
                    self.frames_count = int(nb_frames_str)
                    return self.frames_count
        except (ffmpeg.Error, KeyError, IndexError, AttributeError) as e:
            Log.log_debug(f"ffprobe (nb_frames) for {path_to_probe.name} failed or parse error: {str(e)}")
        except Exception as e:  # General exceptions
            Log.log_debug(f"Unexpected error getting nb_frames for {path_to_probe.name}: {str(e)}")

        # Method 2: duration * frame_rate
        try:
            Log.log_debug(f"获取帧数 (duration*rate): {path_to_probe.name}")
            if self.orig_duration is None:  # Ensure duration for input_file is available
                self.get_duration()
            if self.orig_duration is None or self.orig_duration <= 0:
                Log.log_debug(
                    f"Cannot calculate frames using duration for {path_to_probe.name}: duration is {self.orig_duration}")
                return None

            probe_rate_data = ffmpeg.probe(str(path_to_probe), cmd=FFPROBE_PATH,
                                           select_streams='v:0', show_entries='stream=r_frame_rate')
            if not probe_rate_data or 'streams' not in probe_rate_data or not probe_rate_data['streams']:
                Log.log_debug(f"ffprobe (r_frame_rate) for {path_to_probe.name} returned no valid streams.")
                return None

            r_frame_rate_str = probe_rate_data['streams'][0].get('r_frame_rate')
            if not r_frame_rate_str or not '/' in r_frame_rate_str:
                Log.log_debug(f"Invalid r_frame_rate format for {path_to_probe.name}: {r_frame_rate_str}")
                return None

            num_str, denom_str = r_frame_rate_str.split('/')
            if not num_str or not denom_str or not num_str.isdigit() or not denom_str.isdigit():
                Log.log_debug(f"Invalid r_frame_rate components for {path_to_probe.name}: {r_frame_rate_str}")
                return None

            num, denom = int(num_str), int(denom_str)
            if denom <= 0:
                Log.log_debug(f"Invalid r_frame_rate denominator ({denom}) for {path_to_probe.name}")
                return None

            frame_rate = num / denom
            if frame_rate is None or frame_rate <= 0:
                Log.log_debug(f"Invalid frame_rate ({frame_rate}) for {path_to_probe.name}")
                return None

            self.frames_count = int(self.orig_duration * frame_rate)
            return self.frames_count

        except (ffmpeg.Error, KeyError, IndexError, AttributeError, ValueError) as e:
            Log.log_error(f"估算帧数失败 (duration*rate method) for {path_to_probe.name}: {str(e)}")
        except Exception as e:  # General exceptions
            Log.log_error(f"估算帧数失败 (duration*rate method) for {path_to_probe.name}: {str(e)}")
        if self.frames_count is None:
            Log.log_warning(f"无法获取 {path_to_probe.name} 的帧数。")
        return self.frames_count


def transcode_worker(input_file: Path, output_path: Path, tasks: queue.Queue[TranscodeTask | None], transcode_finished: threading.Event) -> None:
    for index, encode_args in enumerate(ARGS, 1):
        output_file = output_path / f"{input_file.stem}_{encode_args.encoder}{encode_args.ext_name}"
        Log.log_info(f"正在处理 ({index}/{len(ARGS)}): {input_file} -> {output_file}")
        task = TranscodeTask(input_file, output_file)
        if not task.transcode(encode_args):
            Log.log_error(f"转码失败: {input_file.name}")
            tasks.put(task)
            continue
        task.get_compression_rate()
        task.get_duration()
        task.get_bitrate()
        res_status = task.to_status()
        if res_status:
            Log.log_success(res_status)
        tasks.put(task)
    Log.log_info(f"所有视频转码任务已提交，等待评估结果...")
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
                Log.log_info(f"正在评估 VMAF: {task.input_file.name} (队列中剩余 {tasks.qsize()})")
                if task.get_vmaf_score() is not None:
                    res_status = task.to_status()
                    if res_status:
                        Log.log_success(res_status)

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
    trascode_finished = threading.Event()

    transcode_thread = threading.Thread(
        target=transcode_worker,
        name="转码",
        args=(input_path, output_path, tasks, trascode_finished)
    )
    transcode_thread.start()
    evaluate_thread = threading.Thread(
        target=evaluate_worker,
        name="评估",
        args=(tasks, results, output_path / OUTPUT_RESULT_NAME, enable_vmaf, trascode_finished)
    )
    evaluate_thread.start()
    transcode_thread.join()
    evaluate_thread.join()

    Log.log_success(f"所有视频处理完成，结果已保存为 {output_path / OUTPUT_RESULT_NAME}")
    print_results(results)


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            if len(sys.argv) == 3:
                main(sys.argv[1], sys.argv[2])
            else:
                main(sys.argv[1])
        else:
            print("用法: python main.py <输入文件> [输出目录]")
    except KeyboardInterrupt:
        Log.log_error("用户中断")
        sys.exit(130)
