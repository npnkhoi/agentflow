"""Drive a running `streamlit run agentflow/viewer.py` server the way a browser
does, so import errors that only appear under the real command are caught.

Connects to the Streamlit websocket, sends a rerun BackMsg (built with
streamlit's own protobuf classes), and reports whether the app script executed
cleanly or raised.

Usage:
    # with the viewer already serving on 8501
    /home/khoi/miniconda3/envs/ds/bin/python experimental/e0729_probe_viewer_server.py
    /home/khoi/miniconda3/envs/ds/bin/python experimental/e0729_probe_viewer_server.py --port 8502
"""

import argparse
import asyncio

import websockets
from streamlit.proto.BackMsg_pb2 import BackMsg
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg


async def probe(port: int, seconds: float, url: str | None = None) -> int:
    url = url or f"ws://127.0.0.1:{port}/_stcore/stream"
    msg = BackMsg()
    msg.rerun_script.query_string = ""

    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(msg.SerializeToString())

        exceptions, elements = [], 0
        deadline = asyncio.get_event_loop().time() + seconds
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                break
            if isinstance(raw, str):
                continue
            fwd = ForwardMsg()
            fwd.ParseFromString(raw)
            if fwd.WhichOneof("type") == "delta":
                elements += 1
                element = fwd.delta.new_element
                if element.WhichOneof("type") == "exception":
                    exceptions.append(
                        f"{element.exception.type}: {element.exception.message}"
                    )

    print(f"elements rendered: {elements}")
    for e in exceptions:
        print(f"  EXCEPTION: {e}")
    if exceptions:
        print("RESULT: app raised")
        return 1
    if elements == 0:
        print("RESULT: app produced nothing (did the script run?)")
        return 1
    print("RESULT: app rendered cleanly")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8501)
    ap.add_argument("--seconds", type=float, default=8.0, help="how long to collect messages")
    ap.add_argument("--url", default=None, help="full websocket URL, e.g. wss://<sub>.trycloudflare.com/_stcore/stream")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(probe(args.port, args.seconds, args.url)))


if __name__ == "__main__":
    main()
