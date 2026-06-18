"""
CLI entrypoint for ``burr-sk``.

Thin convenience wrapper around the sensorkit service entrypoint at
``burr.hardware.sk.burr_program:main``. Equivalent to running::

    sensorkit service run burr burr.hardware.sk.burr_program:main

but with burr's ``--site``/``--preset``/``--config`` arg handling on top.
Also hosts the ``--bootstrap`` interactive utility.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from loguru import logger

from burr.core.constants import LOCAL_APP_CONFIG_OVERRIDE


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Burr SensorKit service",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c", type=Path, default=None,
        help="Single YAML config file (legacy/standalone). Mutually exclusive with --site/--preset.",
    )
    parser.add_argument(
        "--site", type=Path, default=None,
        help="Site YAML (hardware, lat/lon, SK entity names, notifications). Combine with --preset.",
    )
    parser.add_argument(
        "--preset", type=Path, default=None,
        help="Mode preset YAML (scheduling balance, streaks, plotting). Deep-merged on top of --site.",
    )
    parser.add_argument(
        "--name", "-n", default="burr",
        help="Service name registered with SensorKit",
    )
    parser.add_argument(
        "--restart", action="store_true",
        help="Automatically restart on failure",
    )
    parser.add_argument(
        "--bootstrap", action="store_true",
        help="Interactive utility: discover SK entities + site position from a running agent and emit a site.yaml fragment. Requires --host.",
    )
    parser.add_argument(
        "--host", default=None,
        help="SK agent hostname (for --bootstrap)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="SK webapi port (for --bootstrap)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Write the generated site.yaml fragment to this path (for --bootstrap). If omitted, prints to stdout.",
    )
    args = parser.parse_args()

    if args.bootstrap:
        if not args.host:
            parser.error("--bootstrap requires --host")
        if args.config or args.site or args.preset:
            parser.error("--bootstrap cannot be combined with --config/--site/--preset")
        from burr.cli.sk_bootstrap import run_bootstrap
        return run_bootstrap(args.host, args.port, args.out)

    # Resolve config paths: either --site + --preset (composed) or a
    # single --config (legacy). If nothing is supplied, fall back to
    # LOCAL_APP_CONFIG_OVERRIDE so behavior matches the old default.
    paths: list[Path] = []
    if args.site or args.preset:
        if args.config:
            parser.error("--config cannot be combined with --site/--preset")
        if not args.site or not args.preset:
            parser.error("--site and --preset must be supplied together")
        paths = [args.site, args.preset]
    elif args.config:
        paths = [args.config]
    else:
        paths = [LOCAL_APP_CONFIG_OVERRIDE]

    # On Windows ``:`` appears inside drive-letter paths, so always use
    # ``;`` there. _resolve_config_path() in burr_program handles both.
    sep = ";" if os.name == "nt" else ":"
    os.environ["BURR_CONFIG"] = sep.join(str(p) for p in paths)

    from sensorkit.api.entrypoint import ShutdownSignal, run_services

    from burr.hardware.sk.burr_program import main as burr_main

    exit_code = 0
    try:
        asyncio.run(
            run_services(
                {args.name: burr_main},
                max_restarts=-1 if args.restart else 0,
            )
        )
    except* ShutdownSignal:
        logger.info("burr-sk exiting on shutdown signal")
    except* Exception as eg:
        for exc in eg.exceptions:
            logger.opt(exception=exc).error("burr-sk exited with error")
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
