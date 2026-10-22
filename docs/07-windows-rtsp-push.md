# 从 Windows 推 RTSP

`testdata/7-480p.mp4` 是点播转码出来的文件：H.264 Main、640×640、恒定 24 fps、约 1.6 Mbps、显示顺序为 I B P B P、关键帧间隔 150 帧（约 6.25 秒），另外带一条几乎无声的 AAC。

本机 `config/app.conf` 按无 B 帧的工业相机来拉流：`fflags=nobuffer`、`max_delay_us=100000`、`buffer_size=262144`、`reorder_queue_size=0`、`low_latency_decode=true`。最后一项会让 VPU 立即输出，不再做显示重排序。队列深度是 2，满了丢最旧的帧，显示也不按 PTS 匀速送出。

直接 `-c copy` 把原片推进 mediaMTX，这两边会对不上：B 帧被立即输出打乱，关键帧要 6 秒才来一个，中间一断就停到下一个 IDR。下面这条命令在 Windows 上按真实时间读文件，重编成无 B 帧、大约每秒一个关键帧的 H.264，再用 TCP 推给本机 mediaMTX。

板子上拉的地址是 `rtsp://<Windows 的局域网 IP>:8554/live`。`127.0.0.1` 只在推流那台 Windows 上有效。

## 可直接粘贴的命令

先改 `$video` 和 `$url`。mediaMTX 与 ffmpeg 在同一台机器上时，`$url` 保持 `127.0.0.1`。

```powershell
$video = "D:\video\7-480p.mp4"
$url   = "rtsp://127.0.0.1:8554/live"

ffmpeg -re -stream_loop -1 -i $video -an `
  -c:v libx264 -pix_fmt yuv420p -profile:v baseline `
  -bf 0 -g 24 -keyint_min 24 `
  -preset veryfast -tune zerolatency `
  -x264-params "scenecut=0:repeat-headers=1" `
  -f rtsp -rtsp_transport tcp $url
```

PowerShell 里行尾的 `` ` `` 只是换行续写，ffmpeg 收不到它。它必须是该行最后一个字符，后面不能有空格。停推流用 `Ctrl+C`。

## 输入

| 写法 | 全称 | 这里的作用 |
| --- | --- | --- |
| `-i` | input | 输入文件。`$video` 是 PowerShell 变量，ffmpeg 收到的是展开后的路径 |
| `-re` | read at native frame rate | 按片源时间戳读。24 fps 就每秒送 24 帧。不加的话会把文件尽快灌进 mediaMTX，画面先冲再停 |
| `-stream_loop -1` | stream loop | 文件结束后从头再读。`-1` 表示无限循环。`0` 是不循环，正数是额外循环次数 |
| `-an` | audio none | 丢掉音频。这条 AAC 推给本程序没有用，还会打乱音视频交织 |

输入侧还可以改：

| 写法 | 作用 |
| --- | --- |
| `-stream_loop 0` | 播完就停，ffmpeg 退出 |
| `-ss 30` | 从第 30 秒开始。放在 `-i` 前面是快速跳转，放在后面是解码到该时间点再开始，更慢、更准 |
| `-t 60` | 只读 60 秒 |
| `-r 24` | 强制输出帧率。原片已经是恒定 24 fps，一般不用加；加上会补帧或丢帧 |

## 视频编码

| 写法 | 全称 | 这里的作用 |
| --- | --- | --- |
| `-c:v libx264` | codec，video | 视频编码器选 x264。同义写法是 `-codec:v`、`-vcodec`。`-c:v copy` 不转码，B 帧和 6.25 秒的关键帧间隔会原样留下 |
| `-pix_fmt yuv420p` | pixel format | 像素格式。VPU 和常见播放器都认 4:2:0。`yuv422p`、`yuv444p` 很多硬解会直接拒绝 |
| `-profile:v baseline` | H.264 profile | Baseline 没有 B 帧，也没有 CABAC。本机 VPU 按这种码流解码最稳。`main`、`high` 压缩更好，编码工具也更复杂 |
| `-bf 0` | bframes | B 帧数量为 0。原片是 I B P B P；本机立即输出时不做显示重排序，B 帧会表现为跳、顿 |
| `-g 24` | GOP size，单位是帧 | 最大关键帧间隔。24 fps 时大约 1 秒一个 IDR。原片是 150 帧，约 6.25 秒 |
| `-keyint_min 24` | minimum keyframe interval | 最小关键帧间隔。和 `-g` 设成一样，避免编码器提前插入关键帧 |
| `-preset veryfast` | x264 速度档 | 越快压缩越差、CPU 越轻。从快到慢：`ultrafast`、`superfast`、`veryfast`、`faster`、`fast`、`medium`、`slow`、`slower`、`veryslow`。实时推流用 `ultrafast` 到 `veryfast` |
| `-tune zerolatency` | x264 调优 | 关掉前瞻，把编码时延压低。`film`、`animation`、`fastdecode` 是为画质或软解准备的 |

和关键帧、码率相关、这条命令没写但经常要加的：

| 写法 | 作用 |
| --- | --- |
| `-force_key_frames "expr:gte(t,n_forced*1)"` | 按时间每 1 秒强制一个关键帧，比只靠 `-g` 更固定 |
| `-x264-params "bframes=0"` | 与 `-bf 0` 同一件事，写进 x264 参数串 |
| `-b:v 1600k` | 目标码率。原片大约 1.6 Mbps。不写时由质量和 preset 决定，码率会漂 |
| `-maxrate 2M -bufsize 2M` | 码率上限和码率控制窗口。用来压住突发，避免打满对端很小的接收缓冲 |
| `-crf 23` | 恒定质量。数字越小越清晰、码率越高，常用 18 到 28。它和 `-b:v` 不要同时当主控 |

## x264 私有参数

`-x264-params` 把编码器内部参数用冒号串起来。命令里用了两个：

| 参数 | 作用 |
| --- | --- |
| `scenecut=0` | 关掉场景切换检测。默认检测到镜头切换会提前插关键帧，GOP 就不再是固定 24 帧 |
| `repeat-headers=1` | 每个关键帧前重复 SPS/PPS。中途拉流或重连时，解码器不用再等文件开头那一套参数集。RTSP 推流保持打开 |

同一串里还可以加 `aud=1`（每个访问单元前加 AUD，部分硬解更稳）和 `nal-hrd=cbr`（配合 `-maxrate`、`-bufsize` 做成更稳的恒定码率）。

## 输出

| 写法 | 全称 | 这里的作用 |
| --- | --- | --- |
| `-f rtsp` | format | 输出容器是 RTSP。ffmpeg 作为发布端连到 URL 里的 mediaMTX |
| `-rtsp_transport tcp` | RTSP 传输 | 媒体数据走 TCP。UDP 丢包后 VPU 会缺参考帧，后面一串花屏或卡住。mediaMTX 和本机 `transport=tcp` 对齐 |
| `$url` | 发布地址 | `127.0.0.1:8554` 是本机 mediaMTX 的默认 RTSP 端口，`/live` 是路径名 |

输出侧常见替换：

| 写法 | 作用 |
| --- | --- |
| `-rtsp_transport udp` | 延迟理论上更低。局域网丢包的代价大，和本机配置相反 |
| `-f flv rtmp://127.0.0.1:1935/live` | 改推 RTMP。mediaMTX 默认同路径也能收，本程序只拉 RTSP |
| `-f null -` | 替换掉 `$url` 之后只编码、不推流，用来确认本机编码是否正常 |

`-an` 的对称项是 `-vn`（不要视频）。这条链路只推视频，保持 `-an` 即可。

## 和本机拉流的对应关系

推流保持无 B 帧、约 1 秒一个关键帧、TCP 时，板子继续用现在的低延迟参数：

```text
transport=tcp
max_delay_us=100000
buffer_size=262144
reorder_queue_size=0
low_latency_decode=true
```

如果坚持 `-c copy` 推原片，本机要把 `low_latency_decode` 改成 `false`，`reorder_queue_size` 至少盖住一两帧 B 帧（例如 8），`max_delay_us` 放到 500000 左右。画面会顺一些，延迟也会上去。6.25 秒才一个关键帧，网络一抖仍会停到下一个 IDR。

拉流、硬解和这些选项的实现见 [拉流、硬解和内存](03-memory-path.md)。
