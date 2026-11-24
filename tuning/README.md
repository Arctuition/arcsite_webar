# Deck 照片效果图：一步出图调试

把画布上设计好的 deck 画进房主照片，一次模型调用完成，不再先分析照片、也不需要房主选位置。这里是调 prompt 和选模型用的工具和结果。

## 当前结论（2026-11-23）

- 模型：`gpt-image-2.6-sunburst`（OpenAI `images/edits`，`quality=high`，`input_fidelity=high`）。Gemini 3 Pro / 3.1 Flash 会改机位、换房子或丢 L 形，分数低一截。
- Prompt：`prompts/v4.txt`。
- 输入，按顺序：
  1. 房主照片；
  2. 画布 3D 视图离屏渲染的三张参考图（1280×960 PNG）：俯视（正交，房子在上）、左前斜（方位 -45°、俯角 40°）、右前斜（方位 +45°、俯角 40°）；
  3. 由 deck JSON 生成的几行文字（`describe.py`：尺寸、L 形每段的宽和深、楼梯从哪段下、栏杆、家具清单和位置）；
  4. 风格文案（`styles.json`，来自 Yuki 的 Deck Styles，去掉了房子、尺寸、家具清单和镜头）。
- 画幅：OpenAI 只出 3:2 / 2:3 / 1:1。先把照片用模糊延伸补边到输出比例，出图后按原位置裁回（`providers.py` 的 `_pad_to` / `_crop`），房子不会被裁或拉近。
- 耗时：每张 1–3 分钟，后端要异步出图。
- 安全审核：带图库水印的照片会被 OpenAI 拒。

| 场景 | 结果（6 项各 0–2 分） |
| --- | --- |
| 大斜角后院 | 2 个风格 × 2 次都 12/12；另外 3 个风格也全对 |
| 替换已有 deck | 5 个风格都对：原位替换、L 形、楼梯、家具、风格 |
| 房子正面（带车库） | 8–9：家具和风格对，deck 会压到车库门一角，正面看不出 L 形 |

## 用法

```sh
.venv/bin/python -m tuning.run --prompt v4 --models gpt-image-2.6-sunburst \
    --styles english-country-cottage modern-minimalist --repeat 2
```

- `photos/`（不提交）：测试照片，`<名字>.jpg`。
- `refs/<style>/{top,left-front,right-front}.png`：示例 deck 在各风格下的三张参考图。
- `deck.json`：示例 deck（L 形，热水池 + 两个火盆），画布导出格式。
- `out/<run>/`（不提交）：每格一张图、`run.json`（prompt 原文、耗时、错误）和对比页 `index.html`。
- 密钥只从环境变量 `OPENAI_API_KEY` / `GEMINI_API_KEY` 读，不打印。

## 两步版接口（`app/deck_render.py`）

一步版 `POST /api/deck/render` 保留；画布改用两步：

1. `POST /api/deck/wall` `{photo, deck}`：photo 是画布准备好的 JPEG（长边 2048、质量 92），deck 是引擎导出的 deck（含 `walls`）。服务端用 Gemini 3.8 Flash 标墙（`camera.mark_wall`，不走 `out/walls.json` 缓存）、检测直线、解相机、沿墙放 deck，返回 `{job, width, height, camera, wall, seconds}`；`camera` 原样传给 `AR_Web_RenderDeckPhotoDiagram`。照片里已有 deck 时，立刻在后台去掉它（`inputs.clean_photo`）。解不出相机 422，Gemini 出错 502。
2. 画布用这个相机渲两张透明图：只有 deck（隐藏房子和地面），和带地面的（只隐藏房子）。
3. `POST /api/deck/compose` `{job, deck_png, ground_png, style, model?, prompt?, retry?}`：去掉埋在地下的部分（`inputs.above_ground`），贴到干净照片上，按 `prompt`（默认 `auto`：看得见 deck 地板用 v10k，否则 v10j；两者分别是 v10c、v10e 加上 `camera.stairs_note` 算出的楼梯位置：楼梯伸出画面时说明它从哪边出画、别处不加楼梯，楼梯都在画面里时与 v10c、v10e 一字不差）让 Gemini 3.1 Flash 出图，按 `check` 打分，最多重试 `retry` 次（默认 2）。返回图、每次的分数、`passed`、`cleaned`。job 30 分钟过期（404）。

每次请求的输入输出存在 `out/live/<时间>-two-step/`（photo、clean、deck/ground 渲染、composite、result、`wall.json`、`meta.json`）。
