#!/usr/bin/env python3
"""Live view of the fall switch of a running G1RLQPController, without a GUI client.

Reads the GUI state that mc_rtc publishes (ipc:///tmp/mc_rtc_pub.ipc by default) and prints the labels of the
"G1RLQPController/Fall switch" category: active policy, p against its threshold, kernel, dwell...

  python3 scripts/fall_switch_monitor.py            # live view
  python3 scripts/fall_switch_monitor.py --dump     # print the decoded GUI tree once (debug)
"""

import argparse
import ctypes
import ctypes.util
import struct
import sys
import time

AF_SP = 1
NN_SUB = 2 * 16 + 1  # NN_PROTO_PUBSUB * 16 + 1
NN_SUB_SUBSCRIBE = 1
NN_MSG = ctypes.c_size_t(-1).value
NN_DONTWAIT = 1


# --------------------------------------------------------------------------------------------- minimal msgpack decoder
def unpack(buf, i=0):
    b = buf[i]
    i += 1
    if b <= 0x7F:
        return b, i
    if 0x80 <= b <= 0x8F:
        return _map(buf, i, b & 0x0F)
    if 0x90 <= b <= 0x9F:
        return _array(buf, i, b & 0x0F)
    if 0xA0 <= b <= 0xBF:
        n = b & 0x1F
        return buf[i:i + n].decode("utf-8", "replace"), i + n
    if b >= 0xE0:
        return b - 0x100, i
    simple = {0xC0: None, 0xC2: False, 0xC3: True}
    if b in simple:
        return simple[b], i
    fixed = {0xCA: ">f", 0xCB: ">d", 0xCC: ">B", 0xCD: ">H", 0xCE: ">I", 0xCF: ">Q",
             0xD0: ">b", 0xD1: ">h", 0xD2: ">i", 0xD3: ">q"}
    if b in fixed:
        fmt = fixed[b]
        n = struct.calcsize(fmt)
        return struct.unpack_from(fmt, buf, i)[0], i + n
    sized = {0xD9: (">B", "str"), 0xDA: (">H", "str"), 0xDB: (">I", "str"),
             0xC4: (">B", "bin"), 0xC5: (">H", "bin"), 0xC6: (">I", "bin"),
             0xDC: (">H", "arr"), 0xDD: (">I", "arr"), 0xDE: (">H", "map"), 0xDF: (">I", "map")}
    if b in sized:
        fmt, kind = sized[b]
        n = struct.unpack_from(fmt, buf, i)[0]
        i += struct.calcsize(fmt)
        if kind == "str":
            return buf[i:i + n].decode("utf-8", "replace"), i + n
        if kind == "bin":
            return bytes(buf[i:i + n]), i + n
        if kind == "arr":
            return _array(buf, i, n)
        return _map(buf, i, n)
    ext = {0xD4: 1, 0xD5: 2, 0xD6: 4, 0xD7: 8, 0xD8: 16}
    if b in ext:
        return None, i + 1 + ext[b]
    if b in (0xC7, 0xC8, 0xC9):
        fmt = {0xC7: ">B", 0xC8: ">H", 0xC9: ">I"}[b]
        n = struct.unpack_from(fmt, buf, i)[0]
        return None, i + struct.calcsize(fmt) + 1 + n
    raise ValueError(f"msgpack: unknown byte 0x{b:02x} at {i - 1}")


def _array(buf, i, n):
    out = []
    for _ in range(n):
        v, i = unpack(buf, i)
        out.append(v)
    return out, i


def _map(buf, i, n):
    out = {}
    for _ in range(n):
        k, i = unpack(buf, i)
        v, i = unpack(buf, i)
        out[k if isinstance(k, (str, int, float, bool, type(None))) else str(k)] = v
    return out, i


# ------------------------------------------------------------------------------------------------------ GUI state tree
def find_category(category, path):
    """mc_rtc GUI category: [name, element, ..., [sub-category, ...]]"""
    if not path:
        return category
    if not isinstance(category, list):
        return None
    for child in category[1:]:
        if isinstance(child, list) and child and isinstance(child[0], list):
            for sub in child:
                if isinstance(sub, list) and sub and sub[0] == path[0]:
                    found = find_category(sub, path[1:])
                    if found is not None:
                        return found
    return None


def labels(category):
    """Values of the elements of a category, as {name: value}. Element: [name, type, stack, value, ...]"""
    out = {}
    if not isinstance(category, list):
        return out
    for el in category[1:]:
        if isinstance(el, list) and len(el) >= 2 and isinstance(el[0], str) and isinstance(el[1], int):
            out[el[0]] = el[3] if len(el) > 3 else None
    return out


class Subscriber:
    def __init__(self, uri):
        self.nn = ctypes.CDLL(ctypes.util.find_library("nanomsg") or "libnanomsg.so")
        self.nn.nn_recv.restype = ctypes.c_int
        self.s = self.nn.nn_socket(AF_SP, NN_SUB)
        if self.s < 0:
            sys.exit("nanomsg: cannot create the SUB socket")
        self.nn.nn_setsockopt(self.s, NN_SUB, NN_SUB_SUBSCRIBE, b"", 0)
        if self.nn.nn_connect(self.s, uri.encode()) < 0:
            sys.exit(f"nanomsg: cannot connect to {uri}")

    def latest(self):
        """Last message available (drops the older ones), or None"""
        msg = None
        while True:
            ptr = ctypes.c_void_p()
            n = self.nn.nn_recv(self.s, ctypes.byref(ptr), NN_MSG, NN_DONTWAIT)
            if n < 0:
                return msg
            msg = ctypes.string_at(ptr, n)
            self.nn.nn_freemsg(ptr)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uri", default="ipc:///tmp/mc_rtc_pub.ipc")
    parser.add_argument("--period", type=float, default=0.1, help="refresh period [s]")
    parser.add_argument("--dump", action="store_true", help="print the decoded GUI state once and exit")
    args = parser.parse_args()

    sub = Subscriber(args.uri)
    tty = sys.stdout.isatty()
    last_active = None
    t_wait = time.time()
    while True:
        raw = sub.latest()
        if raw is None:
            if time.time() - t_wait > 2.0:
                print("waiting for the controller (GUI state on %s)..." % args.uri, file=sys.stderr)
                t_wait = time.time()
            time.sleep(args.period)
            continue
        t_wait = time.time()
        state, _ = unpack(raw)
        if args.dump:
            import pprint
            pprint.pprint(state, width=160)
            return
        root = state[2] if isinstance(state, list) and len(state) > 2 else state
        fs = labels(find_category(root, ["G1RLQPController", "Fall switch"]))
        top = labels(find_category(root, ["G1RLQPController"]))
        if not fs:
            print("no 'G1RLQPController/Fall switch' category in the GUI state (try --dump)", file=sys.stderr)
            time.sleep(1.0)
            continue
        active = fs.get("Active policy", "?")
        p = fs.get("p", float("nan"))
        thr = fs.get("p threshold", float("nan"))
        try:
            bar = "#" * int(round(max(0.0, min(1.0, float(p))) * 40))
            mark = int(round(float(thr) * 40))
            bar = (bar.ljust(40)[:mark] + "|" + bar.ljust(40)[mark + 1:])
        except (TypeError, ValueError):
            bar = ""
        line = (f"{top.get('Policy state', '?'):<16} active: {active:<12} p={float(p):.3f} [{bar}] "
                f"thr={float(thr):.3f}  kernel: {fs.get('Inside the limit kernel', '?'):<3} "
                f"dwell {fs.get('Dwell', '?')}/{fs.get('Dwell required', '?')}  "
                f"operator: {fs.get('Held by the operator', '?'):<3} "
                f"switches {fs.get('Switches since arming', '?')} back {fs.get('Hand-backs since arming', '?')}")
        if active != last_active:
            if tty:
                sys.stdout.write("\r\033[K")
            print(f"{time.strftime('%H:%M:%S')} >>> active policy: {active}")
            last_active = active
        if tty:
            sys.stdout.write("\r\033[K" + line)
            sys.stdout.flush()
        else:
            print(line)
        time.sleep(args.period)


if __name__ == "__main__":
    main()
