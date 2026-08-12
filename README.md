# Meteor Video Detector

检测夜空视频中的流星,输出时间轴标注。核心算法:帧间差分 + 移动链追踪 + 直线拟合 + 亮度衰减,四重过滤。

Validated on a 97-minute 1920×1080 phone night-sky video (Aug 2026): 4 meteors detected, zero false positives, cross-verified with an independent pipeline.

## 安装

依赖:`ffmpeg`(PATH 中)、`numpy`、`opencv-python`。

```bash
pip install numpy opencv-python
```

## 用法

```bash
# 全片检测
python3 meteor_detect.py input.mov -o events.json

# 时间段检测 + 关键帧图
python3 meteor_detect.py input.mov --start 0 --end 600 \
    --threshold 25 --keyframes kf/ -o events.json
```

输出 JSON 时间轴:

```json
{
  "n_events": 1,
  "events": [
    {
      "time_start": "17:30.58", "time_end": "17:31.33",
      "duration_s": 0.75, "speed_px_per_frame": 5.34,
      "max_brightness": 191.0, "residual_px": 0.09,
      "abs_frame_start": 25214, "abs_frame_end": 25232
    }
  ]
}
```

## 算法

1. **帧间差分** `D = |I_t - I_{t-1}|` — 静止星空归零,流星产生高亮条纹
2. **连通域提取** — 每帧保留最亮 2 个 D 分量
3. **移动链追踪** — 50px 内、容忍 1 帧间隙,贪心链接
4. **过滤** — 链长≥5、直线残差≤8px、位移≥25px、速度 2.5-60px/帧、密度≥0.6、亮度≥80、且在窗口内**衰减消失**(流星熄灭,飞机/卫星不会)

### 误报特征对照

| 目标 | 特征 | 过滤规则 |
|---|---|---|
| 飞机 | 规律闪烁、持续、不熄灭 | 不衰减 + 速度低 |
| 卫星 | 数分钟缓慢移动 | 不衰减 + 速度低 |
| 地面车灯 | 大面积弥散弱光 | 亮度/阈值 |
| 热像素 | 原地闪烁、无位移 | 位移不足 |
| **流星** | 突然出现、匀速直线、0.2-1.5s 熄灭 | 全部通过 |

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--threshold` | 30 | 帧差阈值(漏检降 20-25,误报升 40-50) |
| `--min-speed` / `--max-speed` | 2.5 / 60 | 排除卫星、飞机、噪点跳变 |
| `--min-brightness` | 80 | 峰值亮度下限 |
| `--min-displacement` | 25 | 净位移下限 |
| `--max-residual` | 8.0 | 直线拟合残差上限 |
| `--keyframes` | — | 输出标注关键帧目录 |

## 与 MetDetPy 的关系

[MetDetPy](https://github.com/LilacMeteorObservatory/MetDetPy) 是优秀的消费级视频流星检测器,但默认配置误报多(实测 97 分钟视频报 1953 个候选,真流星 4 颗)。本工具可作为独立验证器,或用自身算法直接检测。

## 验证记录

IMG_5613.MOV(1920×1080, 24fps, 97.8 分钟,英仙座流星雨期间):
4 颗流星(00:07.5、17:30.3、23:19.0、64:47.0)全部检出;7×90s 随机片段交叉核验无漏报、无误报。

## License

MIT
