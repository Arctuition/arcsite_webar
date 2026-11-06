# arcsite_ar_player

## 接入

```yaml
dependencies:
  arcsite_ar_player:
    git:
      url: https://github.com/Arctuition/arcsite_webar.git
      path: flutter/arcsite_ar_player
      ref: main
```

## 使用

```dart
import 'package:arcsite_ar_player/arcsite_ar_player.dart';

if (ArPlayer.checkSupport())
  ElevatedButton(
    onPressed: () => ArPlayer.launchUrl(usdzUrl), // 或 ArPlayer.launchBytes(usdzBytes)
    child: const Text('AR Preview'),
  ),
```

| 方法                 | 说明                       |
|--------------------|--------------------------|
| checkSupport()     | 能否打开 AR Quick Look，不会抛异常 |
| launchUrl(url)     | 用 URL 打开模型               |
| launchBytes(bytes) | 用本地生成的 USDZ 字节打开模型       |

## 注意

- 模型按真实尺寸显示，禁用缩放。
- launchUrl / launchBytes 直接调用，前面不能有 await，模型 URL 或字节要提前准备好。
- 不支持的环境调用 launch 会抛 UnsupportedError；URL 或字节不合法时抛 ArgumentError。
