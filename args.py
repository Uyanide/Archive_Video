FFMPEG_PATH = "ffmpeg"
FFPROBE_PATH = "ffprobe"

VMAF_N_THREADS = 4  # n_threads passed to libvmaf
VMAF_SAMPLE_FRAMES = 1000  # number of frames to sample
# the actual number of frames libvmaf samples should be between 0 and VMAF_SAMPLE_FRAMES * 2


class EncodeArgs:
    def __init__(self, encoder: str, ext_name: str, params: dict[str, str]) -> None:
        self.encoder: str = encoder
        self.ext_name: str = ext_name
        self.codec_params: dict[str, str] = {**CODEC_GLOBAL_ARGS, **params}


GLOBAL_ARGS: list[str] = [
    "-hide_banner"
]

CODEC_GLOBAL_ARGS: dict[str, str] = {
    "map": "0",
    "map_metadata": "0",
    "map_chapters": "0",
    "c:a": "copy",
    "c:s": "copy",
}

DECODE_GLOBAL_ARGS: dict[str, str] = {
    "hwaccel": "cuda",
}

ARGS: dict[str, EncodeArgs] = {
    "av1_nvenc": EncodeArgs(
        "av1_nvenc", ".mkv",
        {
            "c:v": "av1_nvenc",
            "multipass": "fullres",
            "rc": "vbr",
            # "cq": "32",
            "b:v": "3M",
            "maxrate": "6M",
            "bufsize": "20M",
            "preset": "p7",
            "profile:v": "main10",
            "pix_fmt": "p010le",
        }),
    "hevc_nvenc": EncodeArgs(
        "hevc_nvenc", ".mkv",
        {
            "c:v": "hevc_nvenc",
            "multipass": "fullres",
            # "rc": "vbr_hq",
            "cq": "28",
            # "b:v": "2M",
            # "maxrate": "5M",
            "bufsize": "20M",
            "preset": "p7",
            "profile:v": "main10",
            "pix_fmt": "p010le",
        }),
    "h264_nvenc": EncodeArgs(
        "h264_nvenc", ".mkv",
        {
            "c:v": "h264_nvenc",
            "multipass": "fullres",
            # "rc": "vbr_hq",
            "cq": "23",
            # "b:v": "2M",
            # "maxrate": "5M",
            "bufsize": "20M",
            "preset": "p7",
            "profile:v": "high",
            "pix_fmt": "yuv420p",
        }),
}

if __name__ == "__main__":
    for name, args in ARGS.items():
        print(f"{name}: {args.encoder} {args.ext_name} {args.codec_params}")
