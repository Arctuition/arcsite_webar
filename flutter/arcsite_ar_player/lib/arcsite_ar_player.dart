/// Shows USDZ models in AR Quick Look from Flutter Web on iPhone and iPad.
///
/// The detection and launch logic is the shared `shared/quicklook/quicklook.js`, embedded in this
/// package, so the webar page and this SDK behave the same.
library;

import 'dart:typed_data';

import 'src/player_stub.dart' if (dart.library.js_interop) 'src/player_web.dart' as impl;

/// AR Quick Look player.
///
/// Call [checkSupport] to decide whether to show an AR button, and [launchUrl] or [launchBytes]
/// straight from its tap handler.
abstract final class ArPlayer {
  /// Whether AR Quick Look can open here.
  ///
  /// True only in a Flutter Web build running in Safari on iPhone or iPad (or in iOS Chrome, Edge,
  /// Firefox or DuckDuckGo, which hand the model to Quick Look too). False on every native platform
  /// (Android, iOS, Windows, macOS, Linux), in Android and desktop browsers, and in iOS in-app browsers
  /// such as WeChat's or the Google app's.
  ///
  /// Never throws: where a Content Security Policy keeps the package from installing its inline
  /// script, it returns false.
  static bool checkSupport() => impl.checkSupport();

  /// Opens the USDZ model at [url] in AR Quick Look, at true scale: pinch to zoom is disabled.
  ///
  /// [url] may be absolute or relative to the page, and an `https:`, `blob:` or `data:` URL. Its query
  /// and fragment are kept. The page never downloads it itself (Quick Look does), so the server needs
  /// no CORS headers; it should send `Content-Type: model/vnd.usdz+zip`.
  ///
  /// Call it directly in a tap handler, with no `await` before it: Safari opens Quick Look only in
  /// response to a user gesture.
  ///
  /// Throws an [UnsupportedError] where Quick Look can't open (the message names the reason: `not-web`,
  /// `not-ios` or `quick-look-unavailable`), a [StateError] where a Content Security Policy blocks the
  /// package's inline script, and an [ArgumentError] when [url] is empty or not a valid URL.
  static void launchUrl(String url) => impl.launchUrl(url);

  /// Opens the USDZ model in [usdz] in AR Quick Look, at true scale: pinch to zoom is disabled.
  ///
  /// The bytes are served to Quick Look from a `blob:` URL, which is released on the next
  /// [launchBytes] call. Call it the same way as [launchUrl].
  ///
  /// Throws like [launchUrl], with an [ArgumentError] when [usdz] is empty or not a USDZ (zip) file.
  static void launchBytes(Uint8List usdz) => impl.launchBytes(usdz);
}
