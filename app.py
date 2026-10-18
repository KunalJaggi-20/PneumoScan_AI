import os
import socket
from server import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 0))
    if port == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            port = 5000 if s.connect_ex(("127.0.0.1", 5000)) != 0 else 5001
    print(f"Starting PneumoScan AI server on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
