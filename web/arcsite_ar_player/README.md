# @arctuition/ar-player

## 接入

暂未发布到 npm，先在包目录打包后安装：

```bash
npm pack
npm install ./arctuition-ar-player-0.1.0.tgz
```

## 使用

```js
import { ArPlayer } from '@arctuition/ar-player';

if (ArPlayer.checkSupport()) {
  button.onclick = () => ArPlayer.launchUrl(usdzUrl); // 或 ArPlayer.launchBytes(usdzBytes)
}
```

| 方法                 | 说明                       |
|--------------------|--------------------------|
| checkSupport()     | 能否打开 AR Quick Look，不会抛异常 |
| launchUrl(url)     | 用 URL 打开模型               |
| launchBytes(bytes) | 用本地生成的 USDZ 字节打开模型       |

## 注意

- 模型按真实尺寸显示，禁用缩放。
- launchUrl / launchBytes 直接调用，前面不能有 await，模型 URL 或字节要提前准备好。
- 不支持的环境调用 launch 会抛 Error；URL 或字节不合法时抛 TypeError。
