// Web builds (dart2js and Wasm): calls the shared quicklook.js, installed on first use.
import 'dart:js_interop';
import 'dart:typed_data';

import 'package:web/web.dart' as web;

import 'quicklook_js.g.dart';

@JS('ArcsiteQuickLook')
external _QuickLook? get _installed;

extension type _QuickLook._(JSObject _) implements JSObject {
  external String? get version;
  external String? unsupportedReason();
  external String? urlError(String url);
  external void open(String url);
  external String? usdzError(JSUint8Array bytes);
  external void openData(JSUint8Array bytes);
}

bool _installFailed = false;

/// The embedded quicklook.js, installed if the page doesn't have this version yet; null if it can't be
/// (a Content Security Policy or Trusted Types blocking inline scripts). A failure is not retried.
_QuickLook? _install() {
  if (_installed case final ql? when ql.version == quickLookVersion) return ql;
  if (_installFailed) return null;
  try {
    // An inline script runs as soon as it is inserted, so the API can stay synchronous.
    // Under a nonce-based CSP it runs only with the page's nonce.
    final script = web.HTMLScriptElement();
    final nonce = (web.document.querySelector('script[nonce]') as web.HTMLScriptElement?)?.nonce ?? '';
    if (nonce.isNotEmpty) script.nonce = nonce;
    script.text = quickLookJs;
    (web.document.head ?? web.document.documentElement!).appendChild(script);
    script.remove();
  } catch (_) {
    // Trusted Types rejects the text assignment; the check below reports it
  }
  if (_installed case final ql? when ql.version == quickLookVersion) return ql;
  _installFailed = true;
  return null;
}

/// The installed API where Quick Look can open; otherwise throws, saying why not.
_QuickLook _supported() {
  final ql = _install() ??
      (throw StateError(
          'AR Quick Look script could not be installed; a Content Security Policy may block inline scripts'));
  if (ql.unsupportedReason() case final reason?) {
    throw UnsupportedError('AR Quick Look is not supported here ($reason)');
  }
  return ql;
}

bool checkSupport() {
  final ql = _install();
  return ql != null && ql.unsupportedReason() == null;
}

void launchUrl(String url) {
  final ql = _supported();
  if (ql.urlError(url) case final error?) throw ArgumentError.value(url, 'url', 'Invalid model URL: $error');
  ql.open(url);
}

void launchBytes(Uint8List usdz) {
  final ql = _supported();
  final bytes = usdz.toJS;
  if (ql.usdzError(bytes) case final error?) throw ArgumentError('Invalid USDZ: $error', 'usdz');
  ql.openData(bytes);
}
