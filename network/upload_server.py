"""HTTP upload server."""
import os
import tempfile
import threading
from flask import Flask, request, jsonify, send_from_directory
from urllib.request import urlopen, Request
from urllib.parse import urlparse
from typing import Callable, Dict, Any, Optional
from utils.logging_utils import get_logger

_logger = get_logger(__name__)


_IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.heic', '.gif'}
_AUDIO_EXTS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.aiff'}
_WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")


class UploadServer:
    """Flask-based HTTP server for receiving uploads."""

    def __init__(
        self,
        host: str,
        port: int,
        config: Dict[str, Any],
        on_upload_callback: Callable[[bytes, int, int], None],
        on_audio_callback: Optional[Callable[[str], None]] = None,
        viewer_hub=None,
    ):
        """
        Initialize the upload server.

        Args:
            host: Host address to bind to
            port: Port to listen on
            config: Server configuration
            on_upload_callback: Callback for image uploads (bytes, width, height)
            on_audio_callback: Callback for audio uploads (temp file path)
            viewer_hub: Optional ViewerHub; serves the browser renderer at / and /ws
        """
        self.viewer_hub = viewer_hub
        self.host = host
        self.port = port
        self.config = config
        self._on_upload = on_upload_callback
        self._on_audio = on_audio_callback
        self.app = None
        self.thread = None

        # Live stats updated by the main loop
        self.stats: Dict[str, Any] = {"fps": 0.0, "chunks": 0}

        # Server limits
        server_cfg = config.get("server", {})
        self.max_download_bytes = int(server_cfg.get("max_download_bytes", 25 * 1024 * 1024))
        self.download_timeout = int(server_cfg.get("download_timeout_sec", 15))

    def start(self) -> None:
        """Start the upload server in a background thread."""
        self.app = self._create_app()
        self.thread = threading.Thread(
            target=self._run_server,
            name="upload-server",
            daemon=True
        )
        self.thread.start()
        _logger.info("Upload server listening on http://%s:%d", self.host, self.port)

    def _create_app(self) -> Flask:
        """Create Flask app with routes."""
        app = Flask(__name__)

        @app.get("/health")
        def health():
            return jsonify({"status": "ok"})

        @app.get("/stats")
        def stats():
            return jsonify(self.stats)

        if self.viewer_hub is not None:
            from flask_sock import Sock
            sock = Sock(app)

            @app.get("/")
            def viewer_index():
                return send_from_directory(_WEB_DIR, "index.html")

            @app.get("/web/<path:filename>")
            def viewer_static(filename):
                return send_from_directory(_WEB_DIR, filename)

            @sock.route("/ws")
            def viewer_ws(ws):
                self.viewer_hub.serve(ws)

        @app.post("/post-compost-upload")
        def post_compost_upload():
            data = None
            filename = None
            target_width = None
            target_height = None

            # --- Multipart file upload ---
            if request.content_type and request.content_type.startswith("multipart/form-data"):
                f = request.files.get("file")
                if not f:
                    return jsonify({"ok": False, "error": "missing 'file' field"}), 400
                data = f.read()
                filename = f.filename or ""
                if len(data) > self.max_download_bytes:
                    return jsonify({"ok": False, "error": "file too large"}), 413
                target_width = request.form.get("width", type=int)
                target_height = request.form.get("height", type=int)

            # --- JSON with URL ---
            elif request.is_json:
                payload = request.get_json(silent=True) or {}
                thingie = payload.get("thingie")
                url = None
                if thingie and thingie.get("file_ref"):
                    url = thingie.get("file_ref").get("file_url")
                    at_obj = thingie.get("at")
                    if at_obj:
                        target_width = at_obj.get("width")
                        target_height = at_obj.get("height")

                if not url:
                    return jsonify({"ok": False, "error": "missing 'url'"}), 400

                parsed = urlparse(url)
                if parsed.scheme not in ("http", "https") or not parsed.netloc:
                    return jsonify({"ok": False, "error": "invalid url"}), 400

                filename = os.path.basename(parsed.path)

                try:
                    req = Request(url, headers={"User-Agent": "fuites-compost/1.0"})
                    with urlopen(req, timeout=self.download_timeout) as resp:
                        dl_chunks = []
                        total = 0
                        while True:
                            chunk = resp.read(256 * 1024)
                            if not chunk:
                                break
                            total += len(chunk)
                            if total > self.max_download_bytes:
                                return jsonify({"ok": False, "error": "file too large"}), 413
                            dl_chunks.append(chunk)
                        data = b"".join(dl_chunks)
                except Exception as exc:
                    return jsonify({"ok": False, "error": f"download failed: {exc}"}), 502

            else:
                return jsonify({"ok": False, "error": "expected multipart/form-data or JSON"}), 400

            if not data:
                return jsonify({"ok": False, "error": "empty payload"}), 400

            try:
                ext = os.path.splitext(filename or "")[1].lower()
                if ext in _AUDIO_EXTS:
                    if not self._on_audio:
                        return jsonify({"ok": False, "error": "audio uploads not supported"}), 400
                    tmp = tempfile.NamedTemporaryFile(
                        suffix=ext, prefix="compost_upload_", delete=False
                    )
                    tmp.write(data)
                    tmp.close()
                    self._on_audio(tmp.name)
                elif ext in _IMAGE_EXTS:
                    self._on_upload(data, target_width, target_height)
                else:
                    return jsonify({"ok": False, "error": f"unsupported file type: {ext or '(none)'}"}), 400
                return jsonify({"ok": True}), 200
            except Exception as exc:
                return jsonify({"ok": False, "error": str(exc)}), 500

        return app

    def _run_server(self) -> None:
        """Run Flask server."""
        self.app.run(
            host=self.host,
            port=self.port,
            debug=False,
            use_reloader=False,
            threaded=True
        )
