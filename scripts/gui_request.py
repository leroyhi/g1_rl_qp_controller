#!/usr/bin/env python3
"""Send a request to the mc_rtc GUI server: gui_req.py 'G1RLQPController' 'ARM policy' [json data]"""
import ctypes, ctypes.util, json, sys, time
nn = ctypes.CDLL(ctypes.util.find_library("nanomsg") or "libnanomsg.so")
AF_SP, NN_PUSH = 1, 16 * 5 + 0  # NN_PROTO_PIPELINE=5 -> NN_PUSH = 80
s = nn.nn_socket(AF_SP, NN_PUSH)
assert s >= 0
assert nn.nn_connect(s, b"ipc:///tmp/mc_rtc_rep.ipc") >= 0
cat = sys.argv[1].split("/") if sys.argv[1] else []
data = json.loads(sys.argv[3]) if len(sys.argv) > 3 else None
msg = json.dumps({"category": cat, "name": sys.argv[2], "data": data}).encode() + b"\0"
time.sleep(0.2)
n = nn.nn_send(s, msg, len(msg), 0)
time.sleep(0.2)
print("sent", n, msg[:-1].decode())
nn.nn_close(s)
