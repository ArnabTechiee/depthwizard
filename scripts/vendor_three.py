"""
DepthWizard - vendor Three.js locally.

The viewer currently pulls Three.js from unpkg at page load. That is the last
network dependency in the whole system, and it breaks two things at once: the
PS's requirement for a standalone offline module, and any demo where the venue
wifi is unreliable. Download the two files once and serve them ourselves.

    python scripts/vendor_three.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import requests

VERSION = "0.160.0"
FILES = {
    f"https://unpkg.com/three@{VERSION}/build/three.module.js":
        "three.module.js",
    f"https://unpkg.com/three@{VERSION}/examples/jsm/controls/OrbitControls.js":
        "OrbitControls.js",
    f"https://unpkg.com/three@{VERSION}/examples/jsm/controls/PointerLockControls.js":
        "PointerLockControls.js",
}

def main() -> int:
    out = Path("viewer/vendor")
    out.mkdir(parents=True, exist_ok=True)
    for url, name in FILES.items():
        print(f"  {name} ...", end="", flush=True)
        r = requests.get(url, timeout=120)
        if r.status_code != 200:
            print(f" FAILED (HTTP {r.status_code})")
            print(f"    download manually from {url}", file=sys.stderr)
            return 1
        (out / name).write_bytes(r.content)
        print(f" {len(r.content)/1024:.0f} KB")

    # OrbitControls imports bare 'three'; the importmap resolves that to our
    # local copy, so no rewriting of its source is needed.
    print(f"\n  wrote {out}/")
    print("  now update the importmap in viewer/index.html to:")
    print('    "three": "./vendor/three.module.js"')
    print('    "three/addons/controls/OrbitControls.js": "./vendor/OrbitControls.js"')
    print('    "three/addons/controls/PointerLockControls.js": "./vendor/PointerLockControls.js"')
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
