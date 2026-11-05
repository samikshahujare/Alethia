"""Verify that the Streamlit application starts up and serves traffic."""

import subprocess
import sys
import time
import urllib.request

def verify_server():
    port = 8503
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.headless",
        "true",
        f"--server.port={port}",
        "--browser.serverAddress=127.0.0.1",
        "--server.enableCORS=false",
    ]
    print(f"Launching Streamlit on port {port}...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        url = f"http://127.0.0.1:{port}/_stcore/health"
        connected = False
        for attempt in range(15):
            time.sleep(1.5)
            try:
                with urllib.request.urlopen(url, timeout=3) as resp:
                    if resp.status == 200:
                        content = resp.read().decode("utf-8")
                        print(f"[SUCCESS] Streamlit health endpoint responded: status {resp.status}, content: '{content}'")
                        connected = True
                        break
            except Exception:
                pass

        if not connected:
            print("[FAIL] Could not connect to Streamlit health endpoint within 25 seconds.")
            # Print any stderr
            _, stderr = proc.communicate(timeout=2)
            print(f"Stderr: {stderr}")
            return False
        return True
    finally:
        print("Terminating Streamlit test server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    success = verify_server()
    sys.exit(0 if success else 1)

