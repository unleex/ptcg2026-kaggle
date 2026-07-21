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


class VisServerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Prevent log spamming the console
        return

    def do_GET(self):
        if not os.path.exists(file_path):
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(
                f"Error: {file_path} not found on disk yet.".encode("utf-8")
            )
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                obj = json.load(f)

            payload = (
                json.dumps(obj["steps"][0][0]["visualize"])
                if "steps" in obj
                else json.dumps(obj)
            )
            escaped_payload = html.escape(payload)

            html_content = f"""<!DOCTYPE html>
            <html>
            <head>
                <title>PTCG Bridge Visualizer</title>
                <style>
                    body {{
                        background: #0f172a;
                        color: #f8fafc;
                        font-family: monospace;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        height: 100vh;
                        margin: 0;
                    }}
                    .loader {{
                        border: 4px solid #1e293b;
                        border-top: 4px solid #6366f1;
                        border-radius: 50%;
                        width: 40px;
                        height: 40px;
                        animation: spin 1s linear infinite;
                        margin-bottom: 20px;
                    }}
                    @keyframes spin {{
                        0% {{ transform: rotate(0deg); }}
                        100% {{ transform: rotate(360deg); }}
                    }}
                </style>
            </head>
            <body>
                <div class="loader"></div>
                <div>Syncing state with competition visualizer...</div>
                <form id="visForm" method="POST" action="https://ptcgvis.heroz.jp/Visualizer/Replay/0">
                    <input type="hidden" name="json" value="{escaped_payload}">
                </form>
                <script>document.getElementById('visForm').submit();</script>
            </body>
            </html>"""

            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html_content.encode("utf-8"))

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(f"Server Error reading JSON: {str(e)}".encode("utf-8"))


server = HTTPServer(("", 0), VisServerHandler)
print("--> Persistent Visualizer Server Active!")
print(f"http://localhost:{server.server_port}")
print(f"Monitoring: '{file_path}' (Refresh tab to pull new modifications)")
print("\nPress Ctrl+C to stop.")

try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\nShutting down visualizer server.")
    sys.exit(0)
