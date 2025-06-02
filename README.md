为了归档百 G 录屏文件夹和满足个人其他类似需求写的小玩意 :)

# 准备

1. 安装依赖 `ffmpeg-python` & `prettytable`;
2. 编辑 `args.py` 中的 `FFMPEG_PATH` `FFPROBE_PATH` 匹配本地环境;
3. 编辑 `args.py` 中 `ARGS` 规定的参数与编码方式以满足具体需求。

# 使用

1. archive.py: 归档指定(复数)文件至指定目录，参照 `archive.py -h`;
2. transcode.py: 以所有可能的方式转码指定文件生成多个文件。`transcode.py <输入文件> [输出目录]`。

# 说明

-   个人硬件配置为 RTX4050 & i5 13500HX，ffmpeg 版本 2025-05-21-git-4099d53759-full_build-www.gyan.dev。
-   脚本已配置 `av1_nvenc`、`hevc_nvenc` 和 `h264_nvenc` 编码器，参数仅针对低动态 2k 录屏文件优化，并不适用于更复杂的环境。
-   脚本使用 `libvmaf` 计算视频质量(若启用)，因 `libvmaf` 计算时消耗 CPU 资源较多，而已经配置的 `*_nvenc` 系列编码器均为 GPU 加速，因此脚本使用了双线程流水线设计以最大化利用硬件资源。
