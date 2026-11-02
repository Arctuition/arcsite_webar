# Deck Web AR PoC

Homeowners design a deck in the browser, then view it in AR with no app to install: **AR Quick Look** on iPhone and iPad, Google's **Scene Viewer** on Android.
The local FastAPI server builds a USDZ (iOS) and a GLB (Android) for each request.

Flow:

1. **Style**: rectangle (1 house wall), L-shape wrapping to the right or to the left (2 walls), U-shape (wraps 3 walls), or 45° angled (cut front corners).
2. **Design**: open sides are labeled A, B, C...; each control maps to one or more labels. Plus the height above ground.
3. **Stairs** (optional): fixed 4 ft wide, centered on one labeled side; tap the selected side again or "No stairs" to skip. Angled (45°) sides and sides shorter than 5′ 6″ can't take stairs. Continuing generates the full-size (1:1) model on the server.
4. **Review**, with two actions:
   - **AR Preview**: AR Quick Look on iOS. The link uses `#allowsContentScaling=0`, so pinch-to-zoom is disabled and the deck stays at real size. On Android the same button opens Scene Viewer through an `intent://` link with `resizable=false` (also real size). The intent goes to the Google app (`com.google.android.googlequicksearchbox`, `mode=ar_preferred`), which US Android phones come with and which `<model-viewer>` uses too: it opens straight into AR, and where Google Play Services for AR is missing Scene Viewer offers to install it or shows the model in 3D. Sending the intent to `com.google.ar.core` (`mode=ar_only`) skips the Google app but fails outright without ARCore. Without the Google app the browser falls back to this page with `#ar-unavailable`, and the page says AR isn't available. China-market phones without Google services (tested: Galaxy S24 China) can't run Scene Viewer either way. Scene Viewer loads the model only over HTTPS, so on Android the page must be opened through an HTTPS address (see Run); over plain HTTP the review step says so.
   - **Show in Photo**: the homeowner takes or picks a photo of their house; Gemini adds the deck to it.
     1. A text model (`gemini-3.8-flash`) finds up to 3 places this deck shape could go, by snap kind: a **wall** (rectangle, angled), an **outside corner** where two walls meet (L), or a **part of the house that sticks out**, front plus both sides (U). For each: where it meets the ground in the image, how far the camera is off to its left or right, and which one is recommended. Cached per photo and snap kind, so rectangle and angled share one analysis.
     2. The photo shows the candidates as highlighted bands and the deck as a bar on the recommended one. Tapping snaps to the nearest candidate; on a wall, dragging also slides the deck along it. **Add Deck** generates.
     3. The server renders the generated USDZ from that viewpoint and from the top as plain diagrams, draws the bar on a copy of the photo as a magenta guide line, and sends photo, guide and diagrams, each labeled, with a description of the deck relative to the building. The user never sees these inputs.
     4. Each result is checked in two passes. First without a model: magenta from the guide line left in the image, or the photo changed away from the deck (a coarse grid compared outside the guide area, which catches zoomed, shifted or recomposed results; results that kept the framing measured ≤ 0.10, the limit is 0.2). Then by the text model: one set of stairs where the design puts them, the house unchanged, the framing kept, nothing added besides the deck (no upper decks, pergolas or roofs), no guide line, and the deck at about its designed height. Two results are generated in parallel and the first that passes is shown; if neither does, one more round of two, and failing that the best result is shown with a note saying it may not fully match (one that is only off-design is preferred over one that spoiled the photo). In testing a single generation matched about 40–50% of the time on hard photos and the checker agreed with a human on 13/14, so most photos get a matching result in 35–70 s, at up to 4 generations (~$0.07 each) per photo. Material changes are not checked.
     The two L-shapes share one corner analysis, but each only offers corners that turn its way (the side wall running back on its side); if none does, the photo screen suggests the other L-shape. Nothing is mirrored, so the plan, AR model, references and photo always agree.
     If no wall is found, it goes straight to generating with a straight-on view. In testing, a viewpoint-matched reference put the deck on the right wall with the stairs on the right side where a fixed front view did not; lighter text models mirrored the camera side, so the analysis uses gemini-3.8-flash at low thinking (about 5–20 s, once per photo).
5. **Photo result**: like the iOS ARDeckAI flow. The first result is a design in the chosen material (Natural Cedar, Gray Composite, Ipe Hardwood, White Painted); other materials restyle that result so the deck keeps its shape. Every result stays in a strip to go back to; compare with the original, try again, move the deck (back to step 2), pick a new photo, or save. A **Prompts** panel at the bottom shows the design and restyle prompt templates for tuning: `[PHOTO]`, `[PLACEMENT GUIDE]`, `[MATCHED VIEW]`, `[TOP VIEW]`, `[FRONT VIEW]` place the images, `{deck}` is the generated description of this deck, `{wall}`, `{placement}` and `{viewpoint}` come from the chosen wall (shown in the panel), and `{style}` is the material. Above each prompt are thumbnails of the exact input images it gets (tap for full size). Edits are kept in the browser (localStorage) and sent with the next request; Reset to Default restores the server's templates.

## Run

Python 3.10+ on macOS, Linux or Windows.

macOS / Linux:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 5001 --reload
```

Windows (PowerShell):

```powershell
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\uvicorn app.main:app --host 0.0.0.0 --port 5001 --reload
```

iPhones can use the plain `http://<LAN IP>:5001` address the desktop page shows as a QR code. **Android** needs the page served over HTTPS with a certificate the phone trusts (a self-signed one is not enough), for example through an HTTPS tunnel or a deployment behind TLS; the GLB link is built from the page's own address, so nothing else changes.

For **Show in Photo**, the server needs a Gemini API key. Put it in a `.env` file in the project root (git-ignored) or set it in the environment; it never reaches the browser:

```
GEMINI_API_KEY=...
# optional, defaults to gemini-3.1-flash-image
GEMINI_IMAGE_MODEL=gemini-3.1-flash-image
# optional, defaults to gemini-3.8-flash (photo viewpoint analysis)
GEMINI_TEXT_MODEL=gemini-3.8-flash
```

Open `http://localhost:5001` on the computer. The page shows a QR code with the computer's LAN address; scan it with an iPhone on the same Wi-Fi and use Safari.
Allow incoming connections if the OS firewall asks (on Windows, allow Python on private networks), or the phone can't reach the server.
Set `WEBAR_PUBLIC_URL` to override the address in the QR code (for example when serving behind HTTPS).

## Layout

- `app/deck.py`: shapes (`make_layout`) and procedural geometry: rim joists, joists, beams, posts, footings, decking, railing, stairs with handrails. Meters, Y-up, house side toward -Z. Members that meet are mitered or offset so no faces are coplanar (coplanar overlaps flicker in AR).
- `app/render.py`: software renderer (numpy z-buffer, no GPU) that reads a USDZ back and draws reference images from the top or any viewpoint.
- `app/photo.py` / `app/gemini.py`: Show in Photo. Builds the prompt from the deck parameters, sends photo + references to Gemini.
- `app/usdz.py`: writes the solids to USD meshes (one per material, UsdPreviewSurface) and packages them with `UsdUtils.CreateNewARKitUsdzPackage`.
- `app/gltf.py`: writes the same solids to a binary glTF (GLB) for Scene Viewer: one mesh per material, PBR colors matching the USDZ, flat normals, faces split into triangle fans (they are convex).
- `app/main.py`: API. `POST /api/models` validates the layout, then generates and caches the model in `generated/` by parameter hash; `GET /models/{id}.usdz` and `/models/{id}.glb` serve it as `model/vnd.usdz+zip` and `model/gltf-binary` (the response carries both URLs); `POST /api/photo/analyze?kind=wall|corner|bump` (body: the photo) returns the candidates; `POST /api/models/{id}/photo?style=` (body: the photo, placement in an `X-Placement` header) returns the photo with the deck added; `POST /api/photo/restyle?style=` (body: a result) changes only the deck's material; `GET /api/photo/styles` lists the materials. `GET /api/models/{id}/prompts` returns the default templates and this deck's `{deck}` text; an edited template goes in the `X-Prompt-Template` header (URI-encoded). `GET /models/{id}/ref/{front|top}.png` and `/models/{id}/ref/matched.png?az=&el=` serve the reference images as sent to Gemini; the design response carries the photo analysis in an `X-View-Analysis` header.
- `static/`: the step-by-step mobile UI. Its shape definitions mirror `make_layout` in `app/deck.py`; keep the two in sync.

On macOS you can validate a model with Apple's checker: `usdchecker --arkit generated/<id>.usdz`.

### Troubleshooting AR on Android

On Android, tapping AR Preview starts a check: if the page hasn't gone to the background (which is what happens when Scene Viewer opens) within 3 s, it says "AR didn't open" with a report ID and posts a report to `POST /api/diag`. The server appends it to `generated/diagnostics.jsonl`; the outcome report (`left_page`, `stayed`, or `fallback` when the Google app is missing) goes out at once with the page visibility events, and a second report with the same ID follows with the user agent and in-app browser (WeChat, QQ, …) if any, phone model, whether Chrome's WebXR reports AR support (a sign ARCore is installed), the intent link, and a HEAD probe of the GLB. The same file logs every model download with the requester's user agent: a `.glb` fetch without `probe=1` means Scene Viewer did open and load the model.

## Not yet covered

- Textures, and attaching the deck to a real house wall (Quick Look and Scene Viewer only place on a plane)
- WebXR on Android (Scene Viewer covers the lightweight path)
- TODO (Android, agreed plan): show an in-page 3D view (`<model-viewer>` with the GLB) when AR can't open; verify on a US/international Android phone. China-market phones without Google services are out of scope.
