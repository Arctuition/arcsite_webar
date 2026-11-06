// AR Quick Look player for the web: the same API as the Flutter SDK in flutter/arcsite_ar_player,
// both calling the shared quicklook.js (copied here as quicklook.g.js).
import { create } from "./quicklook.g.js";

let quickLook = null;

// The Quick Look API for this page, created on first use; null outside a browser (server rendering)
function installed() {
  if (!quickLook && typeof window !== "undefined" && typeof document !== "undefined") quickLook = create(window);
  return quickLook;
}

// The API where Quick Look can open; otherwise throws, saying why not
function supported() {
  const ql = installed();
  const reason = ql ? ql.unsupportedReason() : "not-browser";
  if (reason) throw new Error(`AR Quick Look is not supported here (${reason})`);
  return ql;
}

export const ArPlayer = Object.freeze({
  checkSupport() {
    const ql = installed();
    return ql !== null && ql.isSupported();
  },

  launchUrl(url) {
    supported().open(url);
  },

  launchBytes(usdz) {
    const ql = supported();
    if (!(usdz instanceof Uint8Array)) throw new TypeError("Invalid USDZ: expected a Uint8Array");
    ql.openData(usdz);
  },
});
