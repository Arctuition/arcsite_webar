import 'package:arcsite_ar_player/arcsite_ar_player.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

// Flutter Web serves the asset assets/deck.usdz at this URL, relative to the page
const deckUrl = 'assets/assets/deck.usdz';

void main() => runApp(const MaterialApp(title: 'AR Player Example', home: ExamplePage()));

class ExamplePage extends StatefulWidget {
  const ExamplePage({super.key});

  @override
  State<ExamplePage> createState() => _ExamplePageState();
}

class _ExamplePageState extends State<ExamplePage> {
  final supported = ArPlayer.checkSupport();
  final urlField = TextEditingController(text: deckUrl);
  Uint8List? deckBytes;

  @override
  void initState() {
    super.initState();
    // Loaded up front: launchBytes must run straight from the tap, with no await before it
    rootBundle.load('assets/deck.usdz').then((data) => setState(() => deckBytes = data.buffer.asUint8List()));
  }

  @override
  void dispose() {
    urlField.dispose();
    super.dispose();
  }

  /// Runs a launch, showing why it failed. On unsupported platforms the buttons stay enabled to show the error.
  void launch(void Function() run) {
    try {
      run();
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('$e')));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('AR Player Example')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Text(
            supported ? 'AR Quick Look is supported here.' : 'AR Quick Look is not supported here.',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: 24),
          TextField(
            controller: urlField,
            decoration: const InputDecoration(labelText: 'USDZ URL', border: OutlineInputBorder()),
          ),
          const SizedBox(height: 8),
          FilledButton(
            onPressed: () => launch(() => ArPlayer.launchUrl(urlField.text.trim())),
            child: const Text('AR Preview from URL'),
          ),
          const SizedBox(height: 24),
          FilledButton(
            onPressed: deckBytes == null ? null : () => launch(() => ArPlayer.launchBytes(deckBytes!)),
            child: Text(deckBytes == null ? 'Loading bytes...' : 'AR Preview from bytes (${deckBytes!.length} B)'),
          ),
        ],
      ),
    );
  }
}
