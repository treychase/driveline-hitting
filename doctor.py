"""Check that this machine can run the Streamlit app, and say what is missing.

    python doctor.py

A blank page in the browser says nothing about which of the half dozen things
the app needs has gone wrong, so this walks them in order - interpreter,
packages, versions, the C3D archive, and whether GitHub is reachable - and
prints a line per check. Nothing here imports Streamlit's runtime or starts a
server, so it is safe to run while the app is up.
"""

import importlib
import platform
import sys
from pathlib import Path

# Package name, the version the app needs, and why. ``None`` means any version.
REQUIREMENTS = [
    ("streamlit", (1, 29), "the app is written against 1.29 and newer"),
    ("pandas", None, "every table in the project"),
    ("numpy", None, "marker maths"),
    ("plotly", None, "the animated figure"),
    ("sklearn", None, "the exit velocity model"),
    ("ezc3d", None, "reading the raw motion capture"),
    ("requests", None, "fetching the data"),
    ("certifi", None, "certificate verification"),
]

OK, WARN, BAD = "ok  ", "warn", "FAIL"


def version_tuple(text):
    """Leading numeric components of a version string, as ints."""
    parts = []
    for piece in str(text).split(".")[:3]:
        digits = "".join(c for c in piece if c.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def check_packages():
    """Import each requirement and compare its version against the floor."""
    results = []
    for name, minimum, why in REQUIREMENTS:
        try:
            module = importlib.import_module(name)
        except ImportError:
            results.append((BAD, name, f"not installed - needed for {why}"))
            continue
        found = getattr(module, "__version__", "unknown")
        if minimum and version_tuple(found) and version_tuple(found) < minimum:
            wanted = ".".join(str(n) for n in minimum)
            results.append((BAD, name, f"{found}, but {why} (needs {wanted}+)"))
        else:
            results.append((OK, name, str(found)))
    return results


def check_data():
    """Whether the C3D archive and the percentile table are on disk."""
    results = []
    c3d_dir = Path("data") / "c3d"
    swings = list(c3d_dir.glob("*/*.c3d")) if c3d_dir.exists() else []
    if swings:
        results.append((OK, "data/c3d", f"{len(swings)} files"))
    else:
        results.append(
            (WARN, "data/c3d", "empty - the app offers a download button on first run")
        )

    percentiles = Path("data") / "biomech_percentiles.csv"
    results.append(
        (OK, str(percentiles), "present")
        if percentiles.exists()
        else (WARN, str(percentiles), "missing - it gets built on first use")
    )
    return results


def check_network():
    """Whether the published metrics tables are reachable from here."""
    try:
        import requests

        from data_functions import POI_METRICS_URL, TIMEOUT, ca_bundle
    except ImportError as error:
        return [(WARN, "network", f"skipped: {error}")]

    try:
        response = requests.head(
            POI_METRICS_URL, verify=ca_bundle(), timeout=TIMEOUT, allow_redirects=True
        )
        response.raise_for_status()
    except Exception as error:
        return [
            (BAD, "raw.githubusercontent.com", f"{type(error).__name__}: {error}"),
            (
                WARN,
                "certificates",
                f"verifying against {ca_bundle()} - behind a proxy that signs its own, "
                "point REQUESTS_CA_BUNDLE at your CA file",
            ),
        ]
    return [(OK, "raw.githubusercontent.com", f"reachable ({response.status_code})")]


def main():
    print(f"python      {platform.python_version()}  ({sys.executable})")
    print(f"platform    {platform.platform()}")
    print(f"working dir {Path.cwd()}")
    if not Path("streamlit_app.py").exists():
        print(
            f"\n{BAD} streamlit_app.py is not in this directory. Run doctor.py, and the app, "
            "from the repository root."
        )
        return 1
    print()

    results = check_packages() + check_data() + check_network()
    width = max(len(name) for _, name, _ in results)
    for status, name, detail in results:
        print(f"{status}  {name:<{width}}  {detail}")

    failures = [r for r in results if r[0] == BAD]
    print()
    if failures:
        print(f"{len(failures)} check(s) failed. Fix those, then: streamlit run streamlit_app.py")
        return 1
    print("All good. Start the app with: streamlit run streamlit_app.py")
    print("If the browser still shows an empty page, reload it with the cache bypassed")
    print("(ctrl-shift-R, or cmd-shift-R on a Mac) - a stale Streamlit frontend renders blank.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
