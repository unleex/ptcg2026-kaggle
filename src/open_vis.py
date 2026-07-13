#!/usr/bin/env python3
import sys
import json
import html
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <path_to_vis.json>")
    sys.exit(1)

file_path = sys.argv[1]
if not os.path.exists(file_path):
    print(f"Error: File not found: {file_path}")
    sys.exit(1)

with open(file_path, "r", encoding="utf-8") as f:
    obj = json.load(f)

payload = (
    json.dumps(obj["steps"][0][0]["visualize"]) if "steps" in obj else json.dumps(obj)
)
escaped_payload = html.escape(payload)

html_content = f"""<!DOCTYPE html>
<html>
<body>
    <form id="visForm" method="POST" action="https://ptcgvis.heroz.jp/Visualizer/Replay/0">
        <input type="hidden" name="json" value="{escaped_payload}">
    </form>
    <script>document.getElementById('visForm').submit();</script>
</body>
</html>"""


class VisServerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html_content.encode("utf-8"))
        sys.exit(0)  # Exit script immediately after serving the page


print("--> Click the link below to open the visualizer:")
print("http://localhost:8089")
print("\nWaiting for browser connection... (Ctrl+C to cancel)")

try:
    HTTPServer(("", 8089), VisServerHandler).handle_request()
except KeyboardInterrupt:
    sys.exit(0)
