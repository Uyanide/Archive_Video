import ffmpeg
from pathlib import Path
import json
import hashlib
import console_log as log
from args import *
import subprocess


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


def _hash_str(seed: str) -> str:
    """Generate a random file name using SHA-256 hash."""
    return hashlib.sha256(seed.encode('utf-8')).hexdigest()[:16]


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
        if input_file == output_file:
            raise ValueError("输入文件和输出文件不能相同。")
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

        ffmpeg_codec_options = args.codec_params

        try:
            stream = ffmpeg.input(str(self.input_file), **DECODE_GLOBAL_ARGS)
            stream = ffmpeg.output(stream, str(self.output_file), **ffmpeg_codec_options).global_args(*GLOBAL_ARGS)

            log.debug(f"执行命令 (转码): ffmpeg {' '.join(stream.get_args())}")
            stream.run(cmd=FFMPEG_PATH, quiet=True, overwrite_output=True, capture_stderr=True)

            self.isDone = True
            self.encoder = args.encoder
            return True
        except ffmpeg.Error as e:
            error_message = e.stderr.decode('utf-8', errors='replace') if e.stderr else str(e)
            log.error(f"转码失败: {error_message}")
        except Exception as e:
            log.error(f"转码时发生意外错误: {str(e)}")

        self.mark_failed()
        _delete_file(self.output_file)
        return False

    def get_vmaf_score(self, log_dir: Path) -> float | None:
        """Get the VMAF score from the result."""
        if not self.isDone or self.failed:
            return None
        if not self.frames_count:
            self.get_frames_count()
        if not self.frames_count:  # Ensure frames_count is available
            log.warning(f"无法获取 {self.input_file.name} 的帧数, 跳过 VMAF 计算。")
            return None

        if self.vmaf_mean is not None:  # Already calculated
            return self.vmaf_mean

        vmaf_log_path = log_dir / f"{_hash_str(self.output_file.stem)}_vmaf.json"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)

            n_subsample_val = max(self.frames_count // VMAF_SAMPLE_FRAMES, 1)

            distorted = ffmpeg.input(str(self.output_file), **DECODE_GLOBAL_ARGS)
            original = ffmpeg.input(str(self.input_file), **DECODE_GLOBAL_ARGS)

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

            log.debug(f"执行命令 (VMAF): ffmpeg {' '.join(ffmpeg.output(stream, '-', format='null').get_args())}")
            (ffmpeg.output(stream, '-', format='null')
             .global_args(*GLOBAL_ARGS)
             .run(cmd=FFMPEG_PATH, quiet=True, capture_stderr=True))

            if not vmaf_log_path.exists():
                log.error(f"VMAF 日志文件 {vmaf_log_path} 未创建。")
                return None

            with open(vmaf_log_path, "r", encoding="utf-8") as f:
                log_data = json.load(f)
                vmaf_dict = log_data.get("pooled_metrics", {}).get("vmaf", {})
                ret: dict[str, float | None] = {}
                for key in ["min", "max", "mean", "harmonic_mean"]:
                    value = vmaf_dict.get(key)
                    if value is None:
                        log.warning(f"VMAF 日志缺少字段: {key}")
                        ret[key] = None
                    else:
                        try:
                            ret[key] = float(value)
                        except ValueError:
                            log.warning(f"VMAF 字段 {key} 不是数字: {value}")
                            ret[key] = None
                self.vmaf_min = ret.get("min")
                self.vmaf_max = ret.get("max")
                self.vmaf_mean = ret.get("mean")
                self.vmaf_harmonic_mean = ret.get("harmonic_mean")
            return self.vmaf_mean

        except ffmpeg.Error as e:
            error_message = e.stderr.decode('utf-8', errors='replace') if e.stderr else str(e)
            log.error(f"{self.input_file.name} VMAF 对比失败 - {error_message[:200]}")
            return None
        except json.JSONDecodeError as e:
            log.error(f"解析 VMAF 日志失败 ({vmaf_log_path}): {str(e)}")
            return None
        except Exception as e:
            log.error(f"VMAF 计算时发生意外错误: {str(e)}")
            return None
        finally:
            _delete_file(vmaf_log_path)

    def get_duration(self) -> tuple[float | None, float | None]:
        """Get the duration of the original and output video files."""

        def _fetch_duration(path: Path) -> float | None:
            try:
                probe = ffmpeg.probe(str(path), cmd=FFPROBE_PATH, show_entries='format=duration')
                duration_str = probe.get('format', {}).get('duration')
                if duration_str:
                    return float(duration_str)
                log.warning(f"无法从 {path.name} 获取时长信息。")
            except ffmpeg.Error as e:
                error_message = e.stderr.decode('utf-8', errors='replace') if e.stderr else str(e)
                log.error(f"获取视频时长失败 ({path.name}): {error_message}")
            except ValueError:
                log.error(f"视频时长格式无效 ({path.name})")
            except Exception as e:
                log.error(f"获取视频时长时发生意外错误 ({path.name}): {str(e)}")
            return None

        if self.orig_duration is not None and self.dist_duration is not None:
            return self.orig_duration, self.dist_duration
        if self.orig_duration is None:
            self.orig_duration = _fetch_duration(self.input_file)

        if self.dist_duration is None and self.output_file.exists() and self.isDone and not self.failed:
            self.dist_duration = _fetch_duration(self.output_file)
        elif not self.output_file.exists() and self.isDone and not self.failed:
            log.warning(f"输出文件 {self.output_file.name} 不存在, 无法获取时长。")

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
            probe_data = ffmpeg.probe(str(path_to_probe), cmd=FFPROBE_PATH,
                                      select_streams='v:0', show_entries='stream=nb_frames')
            if probe_data and 'streams' in probe_data and probe_data['streams']:
                nb_frames_str = str(probe_data['streams'][0].get('nb_frames', ''))
                if nb_frames_str.isdigit():
                    self.frames_count = int(nb_frames_str)
                    return self.frames_count
        except (ffmpeg.Error, KeyError, IndexError, AttributeError) as e:
            log.debug(f"ffprobe (nb_frames) for {path_to_probe.name} failed or parse error: {str(e)}")
        except Exception as e:  # General exceptions
            log.debug(f"Unexpected error getting nb_frames for {path_to_probe.name}: {str(e)}")

        # Method 2: duration * frame_rate
        try:
            if self.orig_duration is None:  # Ensure duration for input_file is available
                self.get_duration()
            if self.orig_duration is None or self.orig_duration <= 0:
                log.debug(
                    f"Cannot calculate frames using duration for {path_to_probe.name}: duration is {self.orig_duration}")
                return None

            probe_rate_data = ffmpeg.probe(str(path_to_probe), cmd=FFPROBE_PATH,
                                           select_streams='v:0', show_entries='stream=r_frame_rate')
            if not probe_rate_data or 'streams' not in probe_rate_data or not probe_rate_data['streams']:
                log.debug(f"ffprobe (r_frame_rate) for {path_to_probe.name} returned no valid streams.")
                return None

            r_frame_rate_str = probe_rate_data['streams'][0].get('r_frame_rate')
            if not r_frame_rate_str or not '/' in r_frame_rate_str:
                log.debug(f"Invalid r_frame_rate format for {path_to_probe.name}: {r_frame_rate_str}")
                return None

            num_str, denom_str = r_frame_rate_str.split('/')
            if not num_str or not denom_str or not num_str.isdigit() or not denom_str.isdigit():
                log.debug(f"Invalid r_frame_rate components for {path_to_probe.name}: {r_frame_rate_str}")
                return None

            num, denom = int(num_str), int(denom_str)
            if denom <= 0:
                log.debug(f"Invalid r_frame_rate denominator ({denom}) for {path_to_probe.name}")
                return None

            frame_rate = num / denom
            if frame_rate is None or frame_rate <= 0:
                log.debug(f"Invalid frame_rate ({frame_rate}) for {path_to_probe.name}")
                return None

            self.frames_count = int(self.orig_duration * frame_rate)
            return self.frames_count

        except (ffmpeg.Error, KeyError, IndexError, AttributeError, ValueError) as e:
            log.error(f"估算帧数失败 (duration*rate method) for {path_to_probe.name}: {str(e)}")
        except Exception as e:  # General exceptions
            log.error(f"估算帧数失败 (duration*rate method) for {path_to_probe.name}: {str(e)}")
        if self.frames_count is None:
            log.warning(f"无法获取 {path_to_probe.name} 的帧数。")
        return self.frames_count


DEFAULT_ENCODER = "hevc_nvenc"

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        log.error("Usage: python proc.py <input_file> [output_file] [encoder]")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("output.mkv")
    task = TranscodeTask(input_file, output_file)
    args = ARGS.get(sys.argv[3]) if len(sys.argv) > 3 else ARGS.get(DEFAULT_ENCODER)
    if not args:
        log.error(f"Unknown encoder: {sys.argv[3] if len(sys.argv) > 3 else DEFAULT_ENCODER}")
        sys.exit(1)
    output_file = output_file.with_suffix(args.ext_name)
    log.info(f"Transcoding {input_file.name} to {output_file.name} using {args.encoder}...")
    if not task.transcode(args):
        log.error(f"Transcoding failed for {input_file.name}.")
        sys.exit(1)
    task.get_duration()
    task.get_bitrate()
    task.get_frames_count()
    task.get_compression_rate()
    status = task.to_status()
    if status:
        log.success(status)
