// Run from the package directory: npm test
// The Quick Look logic itself is tested in shared/quicklook/quicklook.test.js; these cover the package API.
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import test, { afterEach } from "node:test";
import { generate, source, target } from "../tool/embed.js";

const UA = {
  iphoneSafari: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Mobile/15E148 Safari/604.1",
  androidChrome: "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Mobile Safari/537.36",
};
const USDZ = new Uint8Array([0x50, 0x4b, 0x03, 0x04, 1, 2, 3]);

// A browser just big enough for the package, installed as the window and document globals
function fakeBrowser(ua) {
  const clicks = [];
  function element(tag) {
    const attrs = {};
    const el = {
      tag,
      style: {},
      children: [],
      setAttribute: (k, v) => (attrs[k] = String(v)),
      removeAttribute: (k) => delete attrs[k],
      appendChild: (c) => el.children.push(c),
      removeChild: (c) => el.children.splice(el.children.indexOf(c), 1),
      click: () => clicks.push({ href: attrs.href, download: attrs.download ?? null }),
    };
    if (tag === "a") el.relList = { supports: (t) => t === "ar" };
    return el;
  }
  const win = {
    navigator: { userAgent: ua, platform: "iPhone", maxTouchPoints: 5 },
    document: { baseURI: "https://app.example.com/", body: element("body"), createElement: element },
    URL: { createObjectURL: () => "blob:https://app.example.com/1", revokeObjectURL: () => {} },
    Blob: class {},
  };
  globalThis.window = win;
  globalThis.document = win.document;
  return clicks;
}

// A fresh copy of the package, so each test starts before its first use
let copies = 0;
async function load() {
  return (await import(`../index.js?copy=${++copies}`)).ArPlayer;
}

afterEach(() => {
  delete globalThis.window;
  delete globalThis.document;
});

test("quicklook.g.js is up to date with shared/quicklook/quicklook.js", () => {
  assert.equal(readFileSync(target, "utf8"), generate(readFileSync(source, "utf8")), "run npm run build");
});

test("exports only ArPlayer with the same three methods as the Flutter SDK", async () => {
  const pkg = await import("../index.js");
  assert.deepEqual(Object.keys(pkg), ["ArPlayer"]);
  assert.deepEqual(Object.keys(pkg.ArPlayer), ["checkSupport", "launchUrl", "launchBytes"]);
  assert.ok(Object.isFrozen(pkg.ArPlayer));
});

test("importing installs no global", async () => {
  fakeBrowser(UA.iphoneSafari);
  (await load()).checkSupport();
  assert.equal(globalThis.ArcsiteQuickLook, undefined);
  assert.equal(window.ArcsiteQuickLook, undefined);
});

test("outside a browser checkSupport is false and launching throws", async () => {
  const player = await load();
  assert.equal(player.checkSupport(), false);
  assert.throws(() => player.launchUrl("https://cdn.example.com/m.usdz"), /not supported here \(not-browser\)/);
  assert.throws(() => player.launchBytes(USDZ), /not supported here \(not-browser\)/);
});

test("on iPhone Safari launchUrl opens Quick Look with scaling disabled", async () => {
  const clicks = fakeBrowser(UA.iphoneSafari);
  const player = await load();
  assert.equal(player.checkSupport(), true);
  player.launchUrl("https://cdn.example.com/m.usdz");
  assert.deepEqual(clicks, [{ href: "https://cdn.example.com/m.usdz#allowsContentScaling=0", download: null }]);
});

test("on iPhone Safari launchBytes opens the bytes as a USDZ blob", async () => {
  const clicks = fakeBrowser(UA.iphoneSafari);
  (await load()).launchBytes(USDZ);
  assert.deepEqual(clicks, [{ href: "blob:https://app.example.com/1#allowsContentScaling=0", download: "model.usdz" }]);
});

test("on Android checkSupport is false and launching throws without opening anything", async () => {
  const clicks = fakeBrowser(UA.androidChrome);
  const player = await load();
  assert.equal(player.checkSupport(), false);
  assert.throws(() => player.launchUrl("https://cdn.example.com/m.usdz"), /not supported here \(not-ios\)/);
  assert.throws(() => player.launchBytes("not bytes"), /not supported here \(not-ios\)/);
  assert.equal(clicks.length, 0);
});

test("invalid arguments throw TypeError without opening anything", async () => {
  const clicks = fakeBrowser(UA.iphoneSafari);
  const player = await load();
  assert.throws(() => player.launchUrl(""), TypeError);
  assert.throws(() => player.launchUrl("http://[bad/x.usdz"), TypeError);
  assert.throws(() => player.launchBytes(USDZ.buffer), TypeError);
  assert.throws(() => player.launchBytes([0x50, 0x4b, 0x03, 0x04]), TypeError);
  assert.throws(() => player.launchBytes(new Uint8Array([1, 2, 3, 4])), /not a USDZ/);
  assert.equal(clicks.length, 0);
});
