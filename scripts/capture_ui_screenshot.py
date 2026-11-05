"""Capture a high-fidelity screenshot of the running Streamlit app via Chrome DevTools Protocol."""

import asyncio
import base64
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import websockets


async def cdp_capture():
    port = 8505
    cdp_port = 9223
    screenshot_dir = Path("docs/screenshots").resolve()
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = screenshot_dir / "alethia_ui_running.png"

    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

    # 1. Start Streamlit Server
    cmd_streamlit = [
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
        "--server.enableXsrfProtection=false",
    ]
    print(f"Starting Streamlit on port {port}...")
    streamlit_proc = subprocess.Popen(
        cmd_streamlit, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # 2. Start Chrome with remote debugging
    cmd_chrome = [
        chrome_path,
        "--headless=new",
        "--disable-gpu",
        f"--remote-debugging-port={cdp_port}",
        "--window-size=1280,900",
        "about:blank",
    ]
    print(f"Starting Chrome with CDP on port {cdp_port}...")
    chrome_proc = subprocess.Popen(
        cmd_chrome, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    try:
        # Wait for Streamlit server to be confirmed ready first
        health_url = f"http://127.0.0.1:{port}/_stcore/health"
        print("Waiting for Streamlit Uvicorn server to be ready...")
        server_ready = False
        for _ in range(30):
            time.sleep(1.0)
            try:
                with urllib.request.urlopen(health_url, timeout=2) as r:
                    if r.status == 200:
                        server_ready = True
                        print("Streamlit Uvicorn server is confirmed ready!")
                        break
            except Exception:
                pass

        if not server_ready:
            print("[FAIL] Streamlit server did not become ready.")
            return False

        # Wait for Chrome CDP endpoint to be ready
        cdp_version_url = f"http://127.0.0.1:{cdp_port}/json/version"
        for _ in range(15):
            time.sleep(0.5)
            try:
                with urllib.request.urlopen(cdp_version_url, timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass

        # Create new tab navigated to Streamlit
        target_url = f"http://127.0.0.1:{cdp_port}/json/new?http://127.0.0.1:{port}"
        req = urllib.request.Request(target_url, method="PUT")
        with urllib.request.urlopen(req, timeout=5) as r:
            tab_info = json.loads(r.read().decode("utf-8"))

        ws_url = tab_info.get("webSocketDebuggerUrl")
        print(f"Connected to CDP tab WebSocket: {ws_url}")

        async with websockets.connect(ws_url) as ws:
            # Enable Page and Runtime domain
            await ws.send(json.dumps({"id": 1, "method": "Page.enable"}))
            await ws.recv()
            await ws.send(json.dumps({"id": 2, "method": "Runtime.enable"}))
            await ws.recv()

            # Explicitly navigate to Streamlit URL
            print(f"Explicitly navigating tab to http://127.0.0.1:{port}...")
            await ws.send(json.dumps({"id": 3, "method": "Page.navigate", "params": {"url": f"http://127.0.0.1:{port}"}}))
            await ws.recv()

            # Poll until Streamlit script finishes running (data-test-script-state="notRunning")
            print("Waiting for Streamlit script execution to complete (notRunning)...")
            msg_id = 20
            for _ in range(30):
                msg_id += 1
                await ws.send(
                    json.dumps(
                        {
                            "id": msg_id,
                            "method": "Runtime.evaluate",
                            "params": {
                                "expression": "document.querySelector('[data-testid=\"stApp\"]')?.getAttribute('data-test-script-state')"
                            },
                        }
                    )
                )
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    if data.get("id") == msg_id:
                        script_state = data.get("result", {}).get("result", {}).get("value")
                        print(f"Current script state: {script_state}")
                        break
                if script_state == "notRunning":
                    print("Streamlit script finished execution! UI fully rendered.")
                    break
                await asyncio.sleep(1.5)

            # Give DOM 1 extra second to settle CSS animations
            await asyncio.sleep(1)

            # Capture screenshot
            print("Capturing full viewport screenshot...")
            await ws.send(
                json.dumps(
                    {
                        "id": 2,
                        "method": "Page.captureScreenshot",
                        "params": {"format": "png", "captureBeyondViewport": False},
                    }
                )
            )

            img_b64 = None
            while True:
                msg = await ws.recv()
                data = json.loads(msg)
                if data.get("id") == 2:
                    img_b64 = data["result"]["data"]
                    break
            img_bytes = base64.b64decode(img_b64)

            screenshot_path.write_bytes(img_bytes)
            print(
                f"[SUCCESS] High-fidelity screenshot captured ({len(img_bytes)} bytes): {screenshot_path}"
            )
            return True

    finally:
        print("Cleaning up processes...")
        chrome_proc.terminate()
        streamlit_proc.terminate()
        try:
            chrome_proc.wait(timeout=3)
            out, err = streamlit_proc.communicate(timeout=3)
            print(f"Streamlit stdout: {out}")
            print(f"Streamlit stderr: {err}")
        except Exception:
            chrome_proc.kill()
            streamlit_proc.kill()


if __name__ == "__main__":
    success = asyncio.run(cdp_capture())
    sys.exit(0 if success else 1)
