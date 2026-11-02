// Native platforms (not a web build): AR Quick Look is never available.
import 'dart:typed_data';

UnsupportedError _notWeb() =>
    UnsupportedError('AR Quick Look needs a Flutter Web build running on iPhone or iPad (not-web)');

bool checkSupport() => false;

void launchUrl(String url) => throw _notWeb();

void launchBytes(Uint8List usdz) => throw _notWeb();
