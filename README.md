# Burr

[![CI](https://github.com/zgazak/burr/actions/workflows/ci.yml/badge.svg)](https://github.com/zgazak/burr/actions/workflows/ci.yml)

Taking the grind out of calibration.

Autonomous calibration and observation scheduling for rapid-deploy SDA telescopes.

Burr converts an observatory into a space domain awareness asset within days of deployment. It autonomously collects twilight flats, photometric standards, calibration satellite passes, sky coverage maps, and lunar background measurements -- all scheduled by lighting condition and time-share priority.

Burr is built off of the descriptions in [Gazak et al. 2025, "Rapid Deployment, Calibration, and Training of Optical Observatories for Space Domain Awareness", AMOS Conference](https://amostech.space/year/2025/rapid-deployment-calibration-and-training-of-optical-observatories-for-space-domain-awareness/) -- see [Citation](#citation) below if you use this software.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager (for development installs)

## Install

Install from PyPI (package name `astro-burr`, imports as `burr`):

```sh
pip install astro-burr
```

Optional extras:

```sh
pip install "astro-burr[sk]"      # SensorKit (NATS-based) control
pip install "astro-burr[direct]"  # direct ASCOM hardware control
pip install "astro-burr[senpai]"  # SENPAI image-processing integration
pip install "astro-burr[notify]"  # Slack notifications
```

### Development install

This repo uses [uv](https://docs.astral.sh/uv/). First install uv, then:

```sh
make sync
source .venv/bin/activate
```

## Configuration

Burr separates **where** you observe from **how** you observe:

- A **site file** (`deploy/<site>/site.yaml`) describes the observatory: coordinates, ASCOM/PWI4 hardware addresses, SensorKit entity names, notifications.
- A **mode preset** (`deploy/presets/*.yaml`) describes the observing strategy: which task sources run, exposure ranges, and scheduling balance.

You compose them at run time with `--site` + `--preset`. See [`deploy/example/`](deploy/example/) for a worked example and [`deploy/presets/`](deploy/presets/) for the bundled modes (`sda`, `science`, `survey`).

| Site section   | Purpose                                                    |
| -------------- | --------------------------------------------------------- |
| `site`         | Coordinates and name (timezone auto-detected)             |
| `hardware`     | ASCOM device addresses, ports, active flags               |
| `sk`           | SensorKit entity names (for `burr-sk`)                     |
| `weather`      | Rain/humidity thresholds and recovery timers              |
| `notifications`| Slack channel for status updates                          |
| `runtime`      | Output directory                                          |

### Scheduling strategies

Each task source has a scheduling strategy:

- **`time_share`** -- target percentage of observing time (e.g. photometry 50%, coverage 30%)
- **`interval`** -- minimum minutes between runs (e.g. calsats every 10 min)
- **`one_shot`** -- run once per lighting condition (e.g. twilight flats)

### Task sources

| Source                  | What it collects                                         | Typical window    |
| ----------------------- | -------------------------------------------------------- | ----------------- |
| `twilight_flats`        | Flat field frames with auto-exposure adjustment          | Nautical twilight |
| `calsats`               | Calibration satellite TLE tracking passes                | Twilight + night  |
| `photometric_standards` | Landolt standard star observations with streaks          | Night             |
| `coverage`              | Sparse sky maps (Fibonacci sphere sampling)              | Night             |
| `lunar_background`      | Background measurements at angular separations from moon | Night             |

## Quick Start

### Direct mode (ASCOM hardware control)

```sh
burr-direct --site deploy/example/site.yaml --preset deploy/presets/sda.yaml
```

Connects to mount, camera, dome, and weather sensor via ASCOM Alpaca / PWI4. Runs all night autonomously -- opens dome at twilight, schedules observations by priority, closes dome at dawn or if weather deteriorates.

### Simulation mode

Set hardware devices to `active: false` in your site file (or point them at the bundled ASCOM/PWI4 simulators under [`deploy/`](deploy/)):

```sh
burr-direct --site deploy/example/site.yaml --preset deploy/presets/survey.yaml
```

### SensorKit mode (NATS-based control)

```sh
burr-sk --site deploy/example/site.yaml --preset deploy/presets/sda.yaml
```

Runs as a SensorKit program/controller pair, integrating with the [SensorKit](https://github.com/sensorkit/sensorkit) agent framework for multi-system orchestration. Requires the `sk` extra. `burr-sk --bootstrap --host <agent>` discovers SK entities + site position from a running agent and emits a `site.yaml` fragment.

### Safe shutdown

```sh
burr-shutdown --site deploy/example/site.yaml --preset deploy/presets/sda.yaml
```

Closes dome, slews to a bright star for verification, parks mount, disconnects.

## Project Structure

```
src/burr/
    bootstrap.py              # Explicit engine init (no import-time side effects)
    cli/                      # CLI entrypoints (direct, shutdown, sk)
    core/                     # Config, constants, logging, notifications
    models/                   # Pydantic models (tracking, observation, site, run, hardware)
    task_source/              # Observation generators implementing the TaskSource protocol
    scheduler/                # Shared scheduler (strategies, slots, factory)
    hardware/
        factory.py            # Device creation from config
        direct/               # ASCOM control (runner, executor, weather monitor)
        sk/                   # SensorKit integration (program, controller, tasks)
    run/                      # Run state management and persistence
    utils/                    # Astronomy, ephemeris, SpaceTrack utilities
```

## Credentials

Burr reads credentials from the environment (or a `.env` file), never from config:

- `SPACETRACK_USERNAME` / `SPACETRACK_PASSWORD` -- for SpaceTrack TLE downloads (calsats)
- `SLACK_BOT_TOKEN` -- for the optional Slack notifications (`notify` extra)

## Output

Each night produces a run directory at `{output_dir}/{site}_{YYYYMMDD}/`:

```
example_20251201/
    data/           # FITS files (flats, science frames, darks)
    plots/          # Coverage maps, lighting schedules
    metadata/       # run_state.json (resumable)
    logs/           # Application logs
    catalogs/       # TLE catalogs
```

Runs auto-resume if interrupted. Starting again on the same night picks up where it left off.

## Citation

Burr implements the system described in:

> Gazak, J. Z., Swindle, R., Morales, S., Phelps, M., Iott, K., Blackhurst, E., & Fletcher, J. 2025, "Rapid Deployment, Calibration, and Training of Optical Observatories for Space Domain Awareness", Proceedings of the Advanced Maui Optical and Space Surveillance Technologies (AMOS) Conference. [doi:10.64861/XGPY2184](https://doi.org/10.64861/XGPY2184)

If you use Burr in your research, please cite this paper:

```bibtex
@inproceedings{Gazak2025Burr,
  title     = {Rapid Deployment, Calibration, and Training of Optical Observatories for Space Domain Awareness},
  author    = {Gazak, J. Zachary and Swindle, Ryan and Morales, Sierra and Phelps, Matthew and Iott, Kevin and Blackhurst, Eric and Fletcher, Justin},
  booktitle = {Proceedings of the Advanced Maui Optical and Space Surveillance Technologies (AMOS) Conference},
  year      = {2025},
  doi       = {10.64861/XGPY2184}
}
```

## License

[MIT](LICENSE)
