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

画布分两步调（接口沿用后端的 analyze / render 两个名字；原来的一步版接口已删，一步版只留在上面的 `tuning.run` 里做实验）：

1. `POST /api/deck/analyze` `{photo, deck}`：photo 是画布准备好的 JPEG（长边 2048、质量 92），deck 是引擎导出的 deck（含 `walls`）。服务端用 Gemini 3.8 Flash 标墙（`camera.mark_wall`，不走 `out/walls.json` 缓存）、检测直线、解相机、沿墙放 deck，返回 `{job, width, height, camera, wall, seconds}`；`camera` 原样传给 `AR_Web_RenderDeckPhotoDiagram`。照片里已有 deck 时，立刻在后台去掉它（`inputs.clean_photo`）。解不出相机 422，Gemini 出错 502。
2. 画布用这个相机渲两张透明图：只有 deck（隐藏房子和地面），和带地面的（只隐藏房子）。
3. `POST /api/deck/render` `{job, deck_png, ground_png, style, model?, prompt?, retry?}`：去掉埋在地下的部分（`inputs.above_ground`），贴到干净照片上，按 `prompt`（默认 `auto`：看得见 deck 地板用 v10k，否则 v10j；两者分别是 v10c、v10e 加上 `camera.stairs_note` 算出的楼梯位置：楼梯伸出画面时说明它从哪边出画、别处不加楼梯，楼梯都在画面里时与 v10c、v10e 一字不差）让 Gemini 3.1 Flash 出图，按 `check` 打分，最多重试 `retry` 次（默认 2）。返回图、每次的分数、`passed`、`cleaned`。job 30 分钟过期（404）。

每次请求的输入输出存在 `out/live/<时间>-two-step/`（photo、clean、deck/ground 渲染、composite、result、`wall.json`、`meta.json`）。

## 先去 deck、只润色 deck（p2，默认，2026-11-25）

同样两个接口、同样的两张画布截图，服务端流程换成 `pipeline2.py` 的做法（环境变量 `DECK_FLOW=two-step` 换回上面的旧流程）：

1. **检测 + 去 deck 一次调用**（`pipeline2.detect_clean`，Gemini 3.1 Flash Image）：模型先答 DECK / NONE，有就同时出去掉它的图，没有就沿用原图。3 张照片 × 3 次判断全对。模型偶尔答 DECK 却不回图（今天的本地记录 15 次里有 2 次），这时再问一次；还没图，或检测本身失败而标墙说照片里有旧 deck（`existing_deck`），就改用 `inputs.clean_photo` 去掉它；也失败就 502，不再带着旧 deck 往下走（那样新 deck 会叠在旧 deck 上，贴回时旧 deck 露在四周）。`wall.json` 的 `detect` 记着问了几次（`tries`）和是谁去掉的（`cleaned_by`）。
2. **墙和相机**：原图上标墙和检测并行；有旧 deck 时去 deck 图上也标一次，取线条残差（`fit_cost`）最小、又放得下 deck 的解。`camera.fit_problems` 拦两种：deck 大半出画；deck 在画面里不到 15% 宽（相机解坏了，如墙长 1514 ft）。拦下的换一次标墙重来，仍不行 422。deck 比墙宽时不拦，`camera.place` 让它越过墙端（靠左、居中、靠右里画面露出最多的那个）。
3. **本地合成**（`pipeline2.compose_images`）：截图切掉地下部分，贴到干净照片上，用带地面那张估出 deck 的阴影压暗照片（`shadow_from_ground`），deck 颜色向照片靠一半。
4. **只润色 deck**（`prompts/p2l2.txt`）：模型只改 deck、栏杆、楼梯、家具和它们的阴影，不准加花草、不画地下的墩子。答回的图先对齐到合成图（`register`，前景照片 Gemini 会缩放 3–36%），再只把 deck 和阴影一带贴回干净照片（`paste_back`），其余像素是原照片。没截到 deck 时 `/compose` 直接 422。

画布截图隐藏房子、地面和立柱基础（基础在地下，顶面与地面齐平切不掉，会被画成混凝土墩）。

盲评（3 位评委，每组 6 张，满分 12/14）：只润色 deck 再贴回 8.8 vs 整张润色 7.1 vs 旧流程 6.0；加对齐和禁止加花后 9.2；去掉墩子后 18 张没有一张露出地下部分（之前 18 张全有），总分 10.0 → 11.3。还没解决：模型仍会在 deck 下加花（约一半），火盆常被换成沙发或花盆。

实验命令：

```sh
.venv/bin/python -m tuning.pipeline2 detect               # 1：检测 + 去 deck，out/p2/detect/
.venv/bin/python -m tuning.pipeline2 wall --repeat 3      # 2：原图和去 deck 图各解几次相机，out/p2/wall/
.venv/bin/python -m tuning.pipeline2 compose              # 3：用 engine/p2/ 的截图合成，out/p2/compose/
.venv/bin/python -m tuning.pipeline2 polish --variants l2 --name d3   # 4：每张照片出 6 次，out/p2/polish/d3/
```

`engine/p2/<照片>.{deck,ground,bare}.png` 是引擎按 `out/p2/cameras.json` 选出的相机截的图（deck：隐藏房子、地面、基础；ground：隐藏房子、基础；bare：只留地面）。每次请求的记录在 `out/live/<时间>-p2/`（多一张 `polished.jpg`：贴回前模型的原图）。
