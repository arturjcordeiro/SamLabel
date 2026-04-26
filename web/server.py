"""
SAM Labeler — local server
==========================
Run:  python3 server.py
Then open:  http://localhost:5000   (opens automatically)

The server finds index.html automatically — it works regardless of
your current working directory or where you placed server.py.

Place your SAM ONNX models in the ./models/ folder next to server.py:
  models/encoder.onnx
  models/decoder.onnx

Or pass custom paths:
  python3 server.py --encoder /path/to/encoder.onnx --decoder /path/to/decoder.onnx --port 5000
"""

import argparse
import json
import os
import sys
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# ── Locate the web root ───────────────────────────────────────────────────────
# Always resolve relative to THIS file, never to the caller's cwd.
# Also handles the case where server.py is one level above the web root.

def find_web_root() -> Path:
    base = Path(__file__).resolve().parent
    candidates = [
        base,
        base / "web",
        base / "app",
        base / "frontend",
        base / "public",
        base / "src",
    ]
    for c in candidates:
        if (c / "index.html").exists():
            return c.resolve()
    # Fallback — return the script directory and let the error surface clearly
    return base

WEB_ROOT = find_web_root()

if not (WEB_ROOT / "index.html").exists():
    print(f"[ERROR] Could not find index.html near {Path(__file__).parent}")
    print("        Make sure server.py is in the same folder as index.html.")
    sys.exit(1)

# ── Arguments ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="SAM Labeler local server")
parser.add_argument("--encoder",    default=str(WEB_ROOT / "weights" / "encoder.onnx"))
parser.add_argument("--decoder",    default=str(WEB_ROOT / "weights" / "decoder.onnx"))
parser.add_argument("--port",       type=int, default=5000)
parser.add_argument("--no-browser", action="store_true")
args = parser.parse_args()

ENCODER = Path(args.encoder).resolve()
DECODER = Path(args.decoder).resolve()

# ── Config served to the frontend ─────────────────────────────────────────────
config = {
    "encoder_available": ENCODER.is_file(),
    "decoder_available": DECODER.is_file(),
    "encoder_path":      str(ENCODER),
    "decoder_path":      str(DECODER),
    "encoder_url":       "/models/encoder.onnx" if ENCODER.is_file() else None,
    "decoder_url":       "/models/decoder.onnx" if DECODER.is_file() else None,
    "encoder_size_mb":   round(ENCODER.stat().st_size / 1e6, 1) if ENCODER.is_file() else 0,
    "decoder_size_mb":   round(DECODER.stat().st_size / 1e6, 1) if DECODER.is_file() else 0,
}
CONFIG_BYTES = json.dumps(config).encode()

# ── Print startup banner ──────────────────────────────────────────────────────
print()
print("╔══════════════════════════════════════════════╗")
print("║            SAM Labeler Server                ║")
print("╠══════════════════════════════════════════════╣")
print(f"║  Web root : {str(WEB_ROOT):<35}║")
print(f"║  Encoder  : {'✓  ' + ENCODER.name if config['encoder_available'] else '✗  NOT FOUND — see README':<38}║")
print(f"║  Decoder  : {'✓  ' + DECODER.name if config['decoder_available'] else '✗  NOT FOUND — see README':<38}║")
print(f"║  URL      :  http://localhost:{args.port:<16}║")
print("╚══════════════════════════════════════════════╝")
print()

if not config["encoder_available"] or not config["decoder_available"]:
    print("⚠  One or more models not found.")
    print("   Place encoder.onnx and decoder.onnx in:", WEB_ROOT / "models")
    print("   Or use --encoder / --decoder flags.")
    print("   You can still browse models manually inside the app.")
    print()

# ── HTTP handler ──────────────────────────────────────────────────────────────
class Handler(SimpleHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        # Pass the absolute web root — works regardless of CWD
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_GET(self):
        path = self.path.split("?")[0]   # strip query strings

        # /config.json — dynamic JSON about model availability
        if path == "/config.json":
            self.send_response(200)
            self.send_header("Content-Type",   "application/json")
            self.send_header("Content-Length", str(len(CONFIG_BYTES)))
            self.send_header("Cache-Control",  "no-cache")
            self._cors()
            self.end_headers()
            self.wfile.write(CONFIG_BYTES)
            return

        # /models/encoder.onnx and /models/decoder.onnx
        # Stream from the actual file path (may differ from WEB_ROOT/models/)
        if path == "/models/encoder.onnx" and ENCODER.is_file():
            self._stream_file(ENCODER)
            return
        if path == "/models/decoder.onnx" and DECODER.is_file():
            self._stream_file(DECODER)
            return

        # Everything else — static files from WEB_ROOT
        super().do_GET()

    def _stream_file(self, filepath: Path):
        size = filepath.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type",   "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control",  "public, max-age=86400")
        self._cors()
        self.end_headers()
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")

    def log_message(self, fmt, *args):
        # Suppress noisy ONNX model transfers; keep everything else
        msg = args[0] if args else ""
        if "/models/" in str(msg) and "onnx" in str(msg):
            return
        super().log_message(fmt, *args)

# ── Start ─────────────────────────────────────────────────────────────────────
try:
    server = HTTPServer(("localhost", args.port), Handler)
except OSError as e:
    print(f"[ERROR] Cannot bind to port {args.port}: {e}")
    print(f"        Try a different port:  python3 server.py --port 8080")
    sys.exit(1)

url = f"http://localhost:{args.port}"

if not args.no_browser:
    threading.Timer(0.9, lambda: webbrowser.open(url)).start()

print(f"Serving at {url}  (Ctrl+C to stop)\n")
try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\n[INFO] Server stopped.")
    server.shutdown()
