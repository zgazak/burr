#!/usr/bin/env python3
"""
Example script demonstrating the Sky Coverage Manager.

This script shows how to initialize and use the SkyCoverageManager
for systematic sky coverage observations.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from burr.core.config import initialize_config
from burr.models.run import BurrRun
from burr.task_source.coverage.manager import SkyCoverageManager


def main():
    """Demonstrate the sky coverage manager."""

    # Initialize configuration
    initialize_config()

    # Create a test run
    run = BurrRun.from_config(observation_date=datetime.now(ZoneInfo("UTC")))

    print(f"Created run: {run.run_id}")
    print(f"Site: {run.site.name} ({run.site.latitude:.3f}, {run.site.longitude:.3f})")

    # Initialize coverage manager
    manager = SkyCoverageManager(run)

    # Show initial status
    status = manager.get_status_summary()
    print("\nInitial status:")
    print(f"  NSIDE: {status['nside']}")
    print(f"  Sparse fraction: {status['sparse_fraction']}")
    print(f"  Total maps: {status['total_maps']}")

    # Make coverage offers (initializes sparse maps)
    print("\nMaking coverage offers...")
    manager.make_coverage_offers()

    # Show updated status
    status = manager.get_status_summary()
    print("\nAfter initialization:")
    print(f"  Total maps: {status['total_maps']}")
    print(f"  Current map ID: {status['current_map_id']}")
    print(f"  Pixels planned: {status['total_pixels_planned']}")
    print(f"  Pixels observed: {status['total_pixels_observed']}")
    print(f"  Overall completion: {status['overall_completion']:.1%}")

    # Generate a few example collect requests
    print("\nGenerating example collect requests:")
    for i in range(5):
        request = manager.generate_request(task_id=f"example_task_{i:03d}")

        if request:
            target = request.target
            print(f"  Request {i + 1}: Alt={target.altitude_deg:.1f}°, Az={target.azimuth_deg:.1f}°")
            print(f"    Exposures: {request.exposure_seconds}")
        else:
            print(f"  Request {i + 1}: No suitable target found")
            break

    # Show final status
    status = manager.get_status_summary()
    print("\nFinal status:")
    print(f"  Pixels observed: {status['total_pixels_observed']}")
    print(f"  Overall completion: {status['overall_completion']:.1%}")

    # Check if plots were generated
    plot_dir = run.output_directory / "plots" / "coverage"
    if plot_dir.exists():
        plots = list(plot_dir.glob("*.png"))
        print(f"\nGenerated {len(plots)} coverage plots in {plot_dir}")
        for plot in plots:
            print(f"  {plot.name}")
    else:
        print("\nNo plots generated (plotting may be disabled)")


if __name__ == "__main__":
    main()
