"""Interactive utility: generate a burr ``site.yaml`` from a running SK agent.

Queries the SensorKit webapi (``/entities`` and ``/data/snapshot/<controller>``),
walks the user through confirming each entity mapping, and writes a yaml
fragment containing the ``app.site`` and ``app.sk`` blocks.

Invoked via ``burr-sk --bootstrap --host <agent-host> [--out site.yaml]``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import requests
import yaml

WEBAPI_TIMEOUT_SECONDS = 10


def run_bootstrap(host: str, port: int, out_path: Optional[Path]) -> int:
    base = f"http://{host}:{port}"
    print(f"Connecting to SK webapi at {base} ...")
    try:
        entities = _fetch_json(f"{base}/entities")
    except requests.RequestException as e:
        print(f"ERROR: could not reach {base}: {e}", file=sys.stderr)
        return 1

    if not isinstance(entities, list) or not entities:
        print(f"ERROR: {base}/entities returned no entities", file=sys.stderr)
        return 1

    by_type = _group_by_type(entities)
    _print_entity_table(by_type)

    controllers = by_type.get("controller", [])
    if not controllers:
        print("ERROR: no controller entities found on this agent.", file=sys.stderr)
        return 1

    controller = _pick_from_list(
        "Select the controller burr should run (burr_controller_entity)",
        controllers,
        default_index=0,
    )
    if controller is None:
        return 1

    try:
        snapshot = _fetch_json(f"{base}/data/snapshot/{controller['name']}")
    except requests.RequestException as e:
        print(f"ERROR: snapshot fetch failed for {controller['name']}: {e}", file=sys.stderr)
        return 1

    caps = _find_prop(snapshot, "Capabilities") or {}
    site_pos = _find_prop(snapshot, "SitePosition")
    cap_devices = caps.get("devices", {}) if isinstance(caps, dict) else {}

    devices = by_type.get("device", [])
    device_names = [d["name"] for d in devices]

    print()
    print("Device mappings (mount/camera/dome come from the controller's Capabilities):")
    mount_entity = _confirm_name(
        "mount_entity", cap_devices.get("mount"), device_names, required=True
    )
    camera_entity = _confirm_name(
        "camera_entity", cap_devices.get("camera"), device_names, required=True
    )
    dome_entity = _confirm_name(
        "dome_entity", cap_devices.get("dome"), device_names, required=False
    )

    weather_guess = _guess_weather(devices, exclude={mount_entity, camera_entity, dome_entity})
    weather_entity = _confirm_name(
        "weather_entity", weather_guess, device_names, required=False
    )

    other_controllers = [c["name"] for c in controllers if c["name"] != controller["name"]]
    dome_controller_entity: Optional[str] = None
    if other_controllers:
        print()
        print("Extra controllers detected — usually only set dome_controller_entity if")
        print("you have a separate process driving the dome. Press Enter to leave unset.")
        dome_controller_entity = _confirm_name(
            "dome_controller_entity", None, other_controllers, required=False
        )

    print()
    site_name = _prompt_str("Site name (human label, e.g. 'Example Observatory')", default=controller["name"])

    if site_pos:
        lat = float(site_pos["latitude_degrees"])
        lon = float(site_pos["longitude_degrees"])
        alt_km = float(site_pos["altitude_km"])
        print(f"SitePosition from controller: lat={lat}, lon={lon}, alt={alt_km} km")
        if not _prompt_bool("Accept site position from controller?", default=True):
            lat = _prompt_float("Latitude (deg)", default=lat)
            lon = _prompt_float("Longitude (deg)", default=lon)
            alt_km = _prompt_float("Altitude (km)", default=alt_km)
    else:
        print("No SitePosition published on this controller — enter manually.")
        lat = _prompt_float("Latitude (deg)")
        lon = _prompt_float("Longitude (deg)")
        alt_km = _prompt_float("Altitude (km)", default=0.0)

    doc = {
        "app": {
            "site": {
                "name": site_name,
                "latitude": lat,
                "longitude": lon,
                "altitude_km": alt_km,
            },
            "sk": {
                "mount_entity": mount_entity,
                "camera_entity": camera_entity,
                "dome_entity": dome_entity,
                "weather_entity": weather_entity,
                "burr_controller_entity": controller["name"],
                "dome_controller_entity": dome_controller_entity,
            },
        }
    }

    rendered = yaml.safe_dump(doc, sort_keys=False, default_flow_style=False)
    print()
    print("=== Generated site.yaml fragment ===")
    print()
    print(rendered)

    if out_path is None:
        print("(No --out given; fragment above is what would be written.)")
        return 0

    if out_path.exists():
        if not _prompt_bool(f"{out_path} exists. Overwrite?", default=False):
            print("Aborted; not written.")
            return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered)
    print(f"Wrote {out_path}")
    return 0


def _fetch_json(url: str) -> Any:
    r = requests.get(url, timeout=WEBAPI_TIMEOUT_SECONDS)
    r.raise_for_status()
    return r.json()


def _group_by_type(entities: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for e in entities:
        out.setdefault(e.get("entity_type", "unknown"), []).append(e)
    return out


def _print_entity_table(by_type: dict[str, list[dict]]) -> None:
    print()
    print("=== SensorKit entities ===")
    for kind in ("controller", "program", "device", "generic"):
        items = by_type.get(kind, [])
        if not items:
            continue
        print()
        print(f"  {kind}s:")
        for i, e in enumerate(items, 1):
            online = "online" if e.get("online") else "offline"
            extra = ""
            if kind == "controller":
                cd = (e.get("details") or {}).get("controlled_devices") or []
                if cd:
                    extra = f"  (controls: {', '.join(cd)})"
            elif kind == "device":
                arch = e.get("archetype") or "-"
                extra = f"  [{arch}]"
            print(f"    {i}. {e['name']:<24} {online}{extra}")


def _find_prop(snapshot: Any, prop: str) -> Any:
    """Extract a property value from a /data/snapshot/<entity> payload.

    Snapshot entries are [kind, timestamp, {path, prop}, value].
    """
    if not isinstance(snapshot, list):
        return None
    for entry in snapshot:
        if not isinstance(entry, list) or len(entry) < 4:
            continue
        path_info = entry[2]
        if isinstance(path_info, dict) and path_info.get("prop") == prop:
            return entry[3]
    return None


def _guess_weather(devices: list[dict], exclude: Iterable[Optional[str]]) -> Optional[str]:
    excluded = {e for e in exclude if e}
    for d in devices:
        name = d.get("name", "")
        if name in excluded:
            continue
        arch = (d.get("archetype") or "").lower()
        if "weather" in name.lower() or arch == "weather":
            return name
    return None


def _pick_from_list(label: str, items: list[dict], default_index: int = 0) -> Optional[dict]:
    if len(items) == 1:
        print(f"{label}: auto-picking only candidate {items[0]['name']!r}")
        return items[0]
    print()
    print(label)
    for i, it in enumerate(items, 1):
        print(f"  {i}. {it['name']}")
    while True:
        raw = input(f"Pick [1-{len(items)}] (default {default_index + 1}): ").strip()
        if raw == "":
            return items[default_index]
        try:
            idx = int(raw)
            if 1 <= idx <= len(items):
                return items[idx - 1]
        except ValueError:
            pass
        print("  invalid selection")


def _confirm_name(
    label: str,
    suggestion: Optional[str],
    candidates: list[str],
    required: bool,
) -> Optional[str]:
    """Prompt the user for an entity name, offering ``suggestion`` as default."""
    candidate_str = ", ".join(candidates) if candidates else "<none>"
    while True:
        if suggestion and suggestion in candidates:
            raw = input(f"  {label} [{suggestion}]: ").strip()
            if raw == "":
                return suggestion
        else:
            extra = "" if required else " (Enter = leave unset)"
            raw = input(f"  {label}{extra}: ").strip()
            if raw == "":
                if required:
                    print(f"    {label} is required. Candidates: {candidate_str}")
                    continue
                return None
        if candidates and raw not in candidates:
            print(f"    {raw!r} not in candidates: {candidate_str}")
            continue
        return raw


def _prompt_str(label: str, default: Optional[str] = None) -> str:
    hint = f" [{default}]" if default is not None else ""
    while True:
        ans = input(f"{label}{hint}: ").strip()
        if ans:
            return ans
        if default is not None:
            return default
        print("  required")


def _prompt_float(label: str, default: Optional[float] = None) -> float:
    hint = f" [{default}]" if default is not None else ""
    while True:
        ans = input(f"{label}{hint}: ").strip()
        if ans == "" and default is not None:
            return default
        try:
            return float(ans)
        except ValueError:
            print("  expected a number")


def _prompt_bool(label: str, default: bool) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    ans = input(f"{label} {hint}: ").strip().lower()
    if ans == "":
        return default
    return ans in ("y", "yes")
