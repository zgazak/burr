# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Burr is an astronomical observation control and processing system for managing telescope operations. It handles real-time hardware control (mounts, cameras, domes, weather sensors), observation scheduling, and data processing. Python 3.13+, managed with UV and Hatchling.

## Development Commands

```bash
# Setup
make sync                    # uv sync --all-extras
source .venv/bin/activate

# Run
burr-direct --config <path>  # Main hardware control
burr-shutdown --config <path> # Safe hardware shutdown
burr-sk --config <path>      # SensorKit observations

# Test
uv run python -m pytest tests -ra -q

# Test hardware connections
python -m burr.models.ascom --camera <HOST> <PORT>
python -m burr.models.ascom --monitor <HOST> <PORT> <SLACK_CHANNEL>
```

## Architecture

### Source Layout

All source code lives under `src/burr/`. Deploy configs per-site live in `deploy/` (e.g., `deploy/example/`).

### Key Layers

1. **Configuration** (`core/config.py`): Pydantic-based, YAML-loaded, frozen models. Singleton via `get_config()` / `initialize_config()`. `.env` searched at `BURR_ENV_PATH`, cwd, parent dir, home dir. Note: `WeatherConfig` in `core/config.py` holds monitoring thresholds; `WeatherDeviceConfig` in `models/hardware.py` holds ASCOM device connection info.

2. **Site & Lighting** (`models/site.py`): `SiteMetadata` with auto timezone detection from coordinates. `LightingSchedule` computes twilight windows (civil, nautical, astronomical) using astropy. Lighting conditions gate which tasks can run.

3. **Core Models** (`models/`):
   - `tracking.py` — Single source of truth for `TrackingMode` enum + `Rates`/`RatesFromStreak`/`RatesFromStreakDistribution`
   - `observation.py` — Universal `ObservationRequest` + `Target` discriminated union (`RADecTarget`, `AltAzTarget`, `TLETarget`, `LunarTarget`)
   - `run.py` — `BurrRun` state model. IO operations in `run/state.py` (`RunStateManager`)
   - `hardware.py` — Hardware config models (`WeatherDeviceConfig`, `DomeConfig`, etc.)
   - `ascom.py` — ASCOM device entities

4. **Task Sources** (`task_source/`): Independent observation generators implementing the `TaskSource` protocol (`task_source/protocol.py`). Each returns `ObservationRequest`:
   - `flats/` — Twilight flat fields with exposure optimization
   - `calsats/` — Calibration satellite tracking (46+ known IDs in `core/constants.py`)
   - `photometry/` — Photometric standard star observations (Landolt catalog)
   - `coverage/` — Sparse sky mapping (Fibonacci sphere algorithm)
   - `lunar/` — Lunar background measurements

5. **Scheduler** (`scheduler/`): Shared between direct and SK modes.
   - `scheduler.py` — `Scheduler.select_best(condition)` picks the best task source
   - `strategies.py` — `TimeShareStrategy`, `IntervalStrategy`, `OneShotStrategy`
   - `slot.py` — `TaskSlot` wraps a `TaskSource` with scheduling state
   - `factory.py` — `build_scheduler(config, run, sources)` constructs from config

6. **Hardware Control** (`hardware/`):
   - `factory.py` — `create_devices(config)` → `DeviceSet` (eliminates init duplication)
   - `device_interface.py` — Protocol-based device interfaces (`Mount`, `Camera`, `Dome`)
   - **Direct** (`hardware/direct/`):
     - `runner.py` — Thin orchestrator (~200 lines) using Scheduler + Executor
     - `executor.py` — `DirectTaskExecutor` pattern-matches on Target type
     - `weather.py` — `WeatherMonitor` async loop
     - `fits_writer.py` — FITS filename generation and header building
     - `device.py` — `DirectDeviceControl` ASCOM implementation
   - **SensorKit** (`hardware/sk/`):
     - `program.py` — `BurrProgram` using shared Scheduler
     - `controller.py` — `BurrController` + `DomeController`
     - `tasks.py` — `BurrCollectTask` extends `sk.CollectTask`
     - `service.py` — SK service factory

7. **Run Management** (`run/`): `BurrRun` state machine (initialized → running → paused → completed/failed). JSON persistence with atomic writes. Automatic resume of today's interrupted run. Output directory: `{site}_{YYYYMMDD}/`. `RunStateManager` in `run/state.py` handles NATS and IO.

8. **CLI Entrypoints** (`cli/`): `direct.py`, `shutdown.py`, `sk.py`. Each calls `bootstrap.initialize_engine()` explicitly.

### Initialization

`import burr` does NOT trigger side effects. Each CLI entrypoint calls `bootstrap.initialize_engine()` which loads `.env`, configures astropy, and ensures JPL ephemeris files.

### Data Flow

**Direct Mode**: `cli/direct.py` → `DirectRunner` → `create_devices()` + `build_scheduler()` + `DirectTaskExecutor` + `WeatherMonitor`. Main loop: `scheduler.select_best(condition)` → `source.generate_request()` → `executor.execute(request)`.

**SK Mode**: `cli/sk.py` → `BurrProgram` wraps `build_scheduler()`. SK framework calls `generate()` → `scheduler.select_best()` → `_request_to_burr_task()` → `BurrCollectTask`. `BurrController` handles task via pattern match on `target_type`.

### Async Pattern

Hardware control uses `asyncio`. `DirectRunner.start()` is the main async entry point. Signal handlers (SIGINT, SIGTERM) trigger graceful shutdown with dome-first safety ordering.

### Shutdown Procedure

`burr-shutdown` follows a specific safety sequence: close dome first → slew to bright star → take verification image (should be blank) → park mount → disable motors → disconnect.
