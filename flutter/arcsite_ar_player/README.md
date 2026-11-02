# arcsite_ar_player

Shows USDZ models in **AR Quick Look** from Flutter Web on iPhone and iPad, at true scale (pinch to zoom disabled).

| Where the app runs | `checkSupport()` |
|---|---|
| Flutter Web in Safari on iPhone / iPad (also iOS Chrome, Edge, Firefox, DuckDuckGo) | `true` |
| Flutter Web in an iOS in-app browser (WeChat, Facebook, the Google app, ...) | `false` |
| Flutter Web in an Android or desktop browser | `false` |
| Flutter Android, iOS, Windows, macOS, Linux apps | `false` (the package still compiles there) |

## Add it

```yaml
dependencies:
  arcsite_ar_player:
    git:
      url: https://github.com/Arctuition/webar.git
      path: flutter/arcsite_ar_player
      ref: main   # or a tag
```

No changes to `web/index.html` and no initialization: the package installs its script on first use.

## Use it

```dart
import 'package:arcsite_ar_player/arcsite_ar_player.dart';

if (ArPlayer.checkSupport())
  ElevatedButton(
    onPressed: () => ArPlayer.launchUrl(usdzUrl), // or ArPlayer.launchBytes(usdzBytes)
    child: const Text('AR Preview'),
  ),
```

- `launchUrl(String url)`: an `https:` URL (absolute or relative to the page), or a `blob:` / `data:` URL. Query and fragment are kept. The page never downloads the model itself, Quick Look does, so the server needs no CORS headers. It should send `Content-Type: model/vnd.usdz+zip`; Quick Look's request carries no `Authorization` header, so put any credentials in the URL (a signed URL).
- `launchBytes(Uint8List usdz)`: a USDZ generated in the app. It is served from a `blob:` URL that is released on the next `launchBytes` call.
- **Call either one directly in the tap handler, with no `await` before it**: Safari opens Quick Look only in response to a user gesture. Have the URL or bytes ready before the button is tapped.
- Both throw an `UnsupportedError` where Quick Look can't open (the message names the reason: `not-web`, `not-ios` or `quick-look-unavailable`), and an `ArgumentError` for an empty or invalid URL, or bytes that aren't a USDZ (zip) file.
- The script is installed as an inline `<script>`, with the page's nonce if it has one. Where a Content Security Policy still blocks it, `checkSupport()` returns `false` and the launch calls throw a `StateError`.

## Shared code

The detection and launch logic lives in [`shared/quicklook/quicklook.js`](../../shared/quicklook/quicklook.js), which the webar page loads too. To change it (for example to add Quick Look banner options), edit that file, then from this directory:

```bash
node --test ../../shared/quicklook/quicklook.test.js
dart run tool/embed_js.dart
dart test
dart test -p chrome
```

`dart test` fails if `lib/src/quicklook_js.g.dart` wasn't regenerated.

## Try it on a phone

```bash
cd example
flutter run -d web-server --web-hostname 0.0.0.0 --web-port 8080
```

Open `http://<this Mac's LAN address>:8080` in Safari on an iPhone or iPad. Quick Look works over plain HTTP.
