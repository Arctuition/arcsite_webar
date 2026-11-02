// Run: dart test -p chrome
@TestOn('browser')
library;

import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

import 'package:arcsite_ar_player/arcsite_ar_player.dart';
import 'package:test/test.dart';
import 'package:web/web.dart' as web;

const iphoneChrome = 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 like Mac OS X) AppleWebKit/605.1.15 '
    '(KHTML, like Gecko) CriOS/131.0.6778.73 Mobile/15E148 Safari/604.1';

final usdz = Uint8List.fromList([0x50, 0x4b, 0x03, 0x04, 1, 2, 3]);

void runScript(String js) {
  final script = web.HTMLScriptElement()..text = js;
  web.document.head!.appendChild(script);
  script.remove();
}

/// Records the links Quick Look would be opened with, and keeps the test page from navigating.
List<({String href, String? rel, String? download})> recordClicks() {
  final clicks = <({String href, String? rel, String? download})>[];
  void onClick(web.Event event) {
    final a = event.target as web.HTMLAnchorElement;
    event.preventDefault();
    clicks.add((href: a.href, rel: a.getAttribute('rel'), download: a.getAttribute('download')));
  }

  final listener = onClick.toJS;
  web.document.addEventListener('click', listener);
  addTearDown(() => web.document.removeEventListener('click', listener));
  return clicks;
}

void main() {
  group('in desktop Chrome', () {
    test('Quick Look is not supported', () {
      expect(ArPlayer.checkSupport(), isFalse);
    });

    test('launching throws, naming the reason, and opens nothing', () {
      final clicks = recordClicks();
      expect(() => ArPlayer.launchUrl('https://example.com/m.usdz'),
          throwsA(isA<UnsupportedError>().having((e) => e.message, 'message', contains('not-ios'))));
      expect(() => ArPlayer.launchBytes(usdz), throwsUnsupportedError);
      expect(clicks, isEmpty);
    });

    test('the shared script is installed once', () {
      ArPlayer.checkSupport();
      expect(web.window.has('ArcsiteQuickLook'), isTrue);
    });
  });

  group('posing as iPhone Chrome', () {
    setUpAll(() {
      runScript("Object.defineProperty(navigator, 'userAgent', {get: () => '$iphoneChrome', configurable: true});");
    });

    test('Quick Look is supported', () {
      expect(ArPlayer.checkSupport(), isTrue);
    });

    test('launchUrl opens the model without scaling, keeping the query', () {
      final clicks = recordClicks();
      ArPlayer.launchUrl('https://cdn.example.com/deck.usdz?sig=abc');
      expect(clicks.single.href, 'https://cdn.example.com/deck.usdz?sig=abc#allowsContentScaling=0');
      expect(clicks.single.rel, 'ar');
      expect(clicks.single.download, isNull);
    });

    test('launchBytes opens a blob: URL named as USDZ', () {
      final clicks = recordClicks();
      ArPlayer.launchBytes(usdz);
      expect(clicks.single.href, matches(RegExp(r'^blob:.*#allowsContentScaling=0$')));
      expect(clicks.single.download, 'model.usdz');
    });

    test('launchUrl rejects empty and invalid URLs', () {
      final clicks = recordClicks();
      expect(() => ArPlayer.launchUrl(''), throwsArgumentError);
      expect(() => ArPlayer.launchUrl('  '), throwsArgumentError);
      expect(() => ArPlayer.launchUrl('http://[bad/x.usdz'), throwsArgumentError);
      expect(clicks, isEmpty);
    });

    test('a different ArcsiteQuickLook on the page is replaced by this version', () {
      runScript("window.ArcsiteQuickLook = {version: '0.0.0-old'};");
      expect(ArPlayer.checkSupport(), isTrue);
      final clicks = recordClicks();
      ArPlayer.launchUrl('https://cdn.example.com/deck.usdz');
      expect(clicks.single.href, endsWith('#allowsContentScaling=0'));
    });

    test('launchBytes rejects bytes that are not a USDZ file', () {
      final clicks = recordClicks();
      expect(() => ArPlayer.launchBytes(Uint8List(0)), throwsArgumentError);
      expect(() => ArPlayer.launchBytes(Uint8List.fromList('{"a":1}'.codeUnits)), throwsArgumentError);
      expect(clicks, isEmpty);
    });
  });
}
