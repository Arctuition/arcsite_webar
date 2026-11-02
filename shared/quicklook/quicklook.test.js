// Run: node --test shared/quicklook/quicklook.test.js
const test = require("node:test");
const assert = require("node:assert/strict");
const { create } = require("./quicklook.js");

const UA = {
  iphoneSafari: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Mobile/15E148 Safari/604.1",
  iphoneChrome: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/131.0.6778.73 Mobile/15E148 Safari/604.1",
  iphoneGoogleApp: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) GSA/343.0.695551749 Mobile/15E148 Safari/604.1",
  iphoneWeChat: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 MicroMessenger/8.0.50(0x1800323c) NetType/WIFI Language/zh_CN",
  ipadDesktopMode: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Safari/605.1.15",
  macSafari: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Safari/605.1.15",
  androidChrome: "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Mobile Safari/537.36",
  windowsChrome: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
};

// A browser just big enough for quicklook.js: elements record attributes, children and clicks.
function fakeWindow({ ua, platform = "iPhone", maxTouchPoints = 5, relAr = true } = {}) {
  const clicks = [];
  const revoked = [];
  const blobs = [];
  function element(tag) {
    const attrs = {};
    const el = {
      tag,
      style: {},
      children: [],
      setAttribute: (k, v) => (attrs[k] = String(v)),
      getAttribute: (k) => (k in attrs ? attrs[k] : null),
      removeAttribute: (k) => delete attrs[k],
      appendChild: (c) => el.children.push(c),
      removeChild: (c) => el.children.splice(el.children.indexOf(c), 1),
      click: () => clicks.push({ href: attrs.href, rel: attrs.rel, download: attrs.download ?? null, hasImg: el.children.some((c) => c.tag === "img") }),
    };
    if (tag === "a") el.relList = { supports: (t) => relAr && t === "ar" };
    return el;
  }
  const body = element("body");
  const win = {
    navigator: { userAgent: ua, platform, maxTouchPoints },
    document: { baseURI: "https://app.example.com/deck/", body, createElement: element },
    URL: {
      createObjectURL: (blob) => (blobs.push(blob), `blob:https://app.example.com/${blobs.length}`),
      revokeObjectURL: (u) => revoked.push(u),
    },
    Blob: class {
      constructor(parts, opts) {
        this.parts = parts;
        this.type = opts.type;
      }
    },
  };
  return { win, body, clicks, revoked, blobs };
}

const iphone = (opts) => fakeWindow({ ua: UA.iphoneSafari, ...opts });
const USDZ = new Uint8Array([0x50, 0x4b, 0x03, 0x04, 1, 2, 3]);

test("supported in iPhone Safari", () => {
  assert.equal(create(iphone().win).unsupportedReason(), null);
  assert.equal(create(iphone().win).isSupported(), true);
});

test("supported in iPad Safari reporting a Mac (desktop mode)", () => {
  const { win } = fakeWindow({ ua: UA.ipadDesktopMode, platform: "MacIntel", maxTouchPoints: 5 });
  assert.equal(create(win).isSupported(), true);
});

test("Mac Safari is not iOS, although it supports rel=ar", () => {
  const { win } = fakeWindow({ ua: UA.macSafari, platform: "MacIntel", maxTouchPoints: 0 });
  assert.equal(create(win).unsupportedReason(), "not-ios");
});

test("Android and desktop browsers are not iOS", () => {
  for (const ua of [UA.androidChrome, UA.windowsChrome]) {
    const { win } = fakeWindow({ ua, platform: "Linux armv8l", maxTouchPoints: 5, relAr: false });
    assert.equal(create(win).unsupportedReason(), "not-ios");
  }
});

test("iOS in-app browsers without rel=ar can't open Quick Look", () => {
  assert.equal(create(fakeWindow({ ua: UA.iphoneWeChat, relAr: false }).win).unsupportedReason(), "quick-look-unavailable");
});

test("iOS Chrome is supported even where relList says no", () => {
  assert.equal(create(fakeWindow({ ua: UA.iphoneChrome, relAr: false }).win).isSupported(), true);
});

test("the Google app on iOS is not supported even where relList says yes", () => {
  assert.equal(create(fakeWindow({ ua: UA.iphoneGoogleApp, relAr: true }).win).unsupportedReason(), "quick-look-unavailable");
});

test("buildUrl disables scaling and keeps query and other options", () => {
  const ql = create(iphone().win);
  assert.equal(ql.buildUrl("https://cdn.example.com/m.usdz"), "https://cdn.example.com/m.usdz#allowsContentScaling=0");
  assert.equal(ql.buildUrl("https://cdn.example.com/m.usdz?sig=a%2Fb&exp=1"), "https://cdn.example.com/m.usdz?sig=a%2Fb&exp=1#allowsContentScaling=0");
  assert.equal(ql.buildUrl("https://cdn.example.com/m.usdz#canonicalWebPageURL=https://x.com/"), "https://cdn.example.com/m.usdz#canonicalWebPageURL=https://x.com/&allowsContentScaling=0");
  assert.equal(ql.buildUrl("https://cdn.example.com/m.usdz#allowsContentScaling=1&a=b"), "https://cdn.example.com/m.usdz#a=b&allowsContentScaling=0");
  assert.equal(ql.buildUrl("https://cdn.example.com/m.usdz#"), "https://cdn.example.com/m.usdz#allowsContentScaling=0");
});

test("buildUrl resolves relative URLs against the page", () => {
  const ql = create(iphone().win);
  assert.equal(ql.buildUrl("models/m.usdz"), "https://app.example.com/deck/models/m.usdz#allowsContentScaling=0");
  assert.equal(ql.buildUrl("/models/m.usdz"), "https://app.example.com/models/m.usdz#allowsContentScaling=0");
});

test("open clicks a rel=ar link with an image, without download for http(s)", () => {
  const { win, body, clicks } = iphone();
  create(win).open("https://cdn.example.com/m.usdz");
  assert.deepEqual(clicks, [{ href: "https://cdn.example.com/m.usdz#allowsContentScaling=0", rel: "ar", download: null, hasImg: true }]);
  assert.equal(body.children.length, 1);
  assert.equal(body.children[0].children.length, 0, "image removed after the click");
});

test("open reuses one link and sets download only for blob: and data:", () => {
  const { win, body, clicks } = iphone();
  const ql = create(win);
  ql.open("blob:https://app.example.com/1234");
  ql.open("https://cdn.example.com/m.usdz");
  ql.open("data:model/vnd.usdz+zip;base64,UEsDBA==");
  assert.equal(body.children.length, 1);
  assert.deepEqual(clicks.map((c) => c.download), ["model.usdz", null, "model.usdz"]);
  assert.equal(clicks[0].href, "blob:https://app.example.com/1234#allowsContentScaling=0");
});

test("open throws where Quick Look is unavailable, without clicking", () => {
  const { win, clicks } = fakeWindow({ ua: UA.androidChrome, platform: "Linux", relAr: false });
  assert.throws(() => create(win).open("https://cdn.example.com/m.usdz"), /not supported here \(not-ios\)/);
  assert.equal(clicks.length, 0);
});

test("usdzError accepts zip bytes and rejects the rest", () => {
  const ql = create(iphone().win);
  assert.equal(ql.usdzError(USDZ), null);
  assert.match(ql.usdzError(new Uint8Array()), /empty/);
  assert.match(ql.usdzError(new Uint8Array([0x7b, 0x22, 0x61, 0x22])), /not a USDZ/);
  assert.match(ql.usdzError(null), /Uint8Array/);
});

test("openData opens a typed blob and releases the previous one only on the next call", () => {
  const { win, clicks, revoked, blobs } = iphone();
  const ql = create(win);
  ql.openData(USDZ);
  assert.equal(blobs[0].type, "model/vnd.usdz+zip");
  assert.deepEqual(revoked, []);
  assert.equal(clicks[0].href, "blob:https://app.example.com/1#allowsContentScaling=0");
  assert.equal(clicks[0].download, "model.usdz");
  ql.openData(USDZ);
  assert.deepEqual(revoked, ["blob:https://app.example.com/1"]);
});

test("openData rejects invalid bytes before creating a blob", () => {
  const { win, blobs, clicks } = iphone();
  assert.throws(() => create(win).openData(new Uint8Array([1, 2, 3, 4])), TypeError);
  assert.equal(blobs.length, 0);
  assert.equal(clicks.length, 0);
});

test("urlError rejects empty and unparseable URLs", () => {
  const ql = create(iphone().win);
  assert.equal(ql.urlError("https://cdn.example.com/m.usdz"), null);
  assert.equal(ql.urlError(""), "empty");
  assert.equal(ql.urlError("   "), "empty");
  assert.equal(ql.urlError(undefined), "empty");
  assert.equal(ql.urlError("http://[bad/x.usdz"), "not a valid URL");
});

test("open throws on an empty URL instead of opening the page itself", () => {
  const { win, clicks } = iphone();
  assert.throws(() => create(win).open(""), TypeError);
  assert.equal(clicks.length, 0);
});
