@TestOn('vm')
library;

import 'dart:typed_data';

import 'package:arcsite_ar_player/arcsite_ar_player.dart';
import 'package:test/test.dart';

import '../tool/embed_js.dart' as embed;

void main() {
  test('native platforms are not supported', () {
    expect(ArPlayer.checkSupport(), isFalse);
  });

  test('launching on a native platform throws', () {
    final notWeb = throwsA(isA<UnsupportedError>().having((e) => e.message, 'message', contains('not-web')));
    expect(() => ArPlayer.launchUrl('https://example.com/m.usdz'), notWeb);
    expect(() => ArPlayer.launchBytes(Uint8List.fromList([0x50, 0x4b, 3, 4])), notWeb);
  });

  test('the embedded quicklook.js matches shared/quicklook/quicklook.js', () {
    expect(embed.target.readAsStringSync(), embed.generate(embed.source.readAsStringSync()),
        reason: 'run dart run tool/embed_js.dart');
  });
}
