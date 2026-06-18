# Burr Run Management System

A comprehensive system for managing astronomical observation runs with data persistence, state tracking, and resumability, integrated with the existing burr configuration system.

## Key Features

### 🔄 **Run State Management**
- Complete run state persistence using JSON with Pydantic model serialization
- Automatic state saving after every operation
- Resume interrupted runs from exact stopping point
- Track run progress, status, and processing stages

### 📁 **Organized Output Structure**
- Automatic output directory creation based on site and local midnight date
- Structured subdirectories for different data types:
  ```
  {site_name}_{YYYYMMDD}/
  ├── plots/              # Elevation plots, schedules
  ├── calendars/           # Star catalogs, reference data  
  ├── calibration/        # Site calibration data
  ├── processing/         # Intermediate processing files
  ├── logs/              # Run logs and debug info
  └── metadata/          # Run state JSON
  ```

### 📋 **Command History & Logging**
- Complete command execution history with timestamps
- Success/failure tracking with error messages
- Processing stage tracking
- File generation tracking

### ⚙️ **Integrated Configuration**
- Uses the existing `burr.core.config` system
- YAML-based configuration with validation
- Runtime settings integrated with run management
- Plotting configuration automatically applied

## CLI Commands

### 🔄 **burr-direct**
Start the main direct hardware control system:
```bash
burr-direct --config config.yaml
```

### 🛑 **burr-shutdown** 
Safely shutdown all hardware devices:
```bash
burr-shutdown --config config.yaml
```

This command performs a complete hardware shutdown procedure:
1. Connects to hardware devices
2. **Closes the dome first** (safest order)
3. **Slews to bright star & takes verification image** (should be blank if dome closed)
4. Parks the mount
5. Disables mount motors  
6. Disconnects all devices

**Use cases:**
- After stopping the main control system with Ctrl-C
- Emergency shutdown situations
- Manual shutdown when needed
- Before powering off hardware

**Key improvements:**
- ✅ **Dome-first shutdown** prevents accidental light exposure
- ✅ **Verification image** confirms dome closure (blank = success)  
- ✅ **No redundant parking** prevents double-park failures
- ✅ **Emergency disconnect** handles timeout gracefully

### 📡 **burr-sk**
Run SensorKit-based observations:
```bash
burr-sk --config config.yaml
```

### 🐍 **Python Module Usage**
You can also run commands directly:
```bash
python -m burr.run.direct --config config.yaml
python -m burr.run.shutdown --config config.yaml
python -m burr.run.sk --config config.yaml
```

## Usage Example

```python
from burr.run import create_burr_run
from burr.models.site import SiteMetadata
from burr.core.config import initialize_config
from datetime import datetime

# Initialize configuration (usually done once)
initialize_config("config.yaml")

# Define observation site
site = SiteMetadata(
    name="Mauna Kea Observatory",
    latitude=19.8263,
    longitude=-155.4781,
    altitude_km=4.207,
    timezone="Pacific/Honolulu"
)

# Create new run (uses config automatically)
observation_date = datetime(2025, 6, 10, 12, 0, 0)
run = create_burr_run(site, observation_date)

# Run operations with automatic state saving
from burr.run import update_progress, log_command, add_generated_file

update_progress(run, "calibration", 0.3, "running")
log_command(run, "telescope alignment", "Successfully aligned")

# Add generated files
add_generated_file(run, Path("output/plot.png"))
```

## Resuming Runs

```python
from burr.run import find_existing_runs, load_run_state

# Find existing runs (uses config automatically)
existing_runs = find_existing_runs()

# Load most recent run
if existing_runs:
    run = load_run_state(existing_runs[0])
    print(f"Resumed run: {run.run_id}")
    print(f"Progress: {run.progress*100:.1f}%")
    
    # Continue processing...
    update_progress(run, "analysis", run.progress + 0.1)
```

## Configuration Integration

The system integrates with your existing `config.yaml`:

```yaml
app:
  runtime:
    output_dir: "./observations"           # Where runs are stored
    max_concurrent_processes: 4
    enable_parallel_processing: true
    max_retries: 3
    continue_on_error: false
    save_intermediate_files: false
    
  plotting:
    generate_plots: true                   # Auto-generate lighting plots
    plot_format: "png"                     # png, pdf, svg
    plot_dpi: 150
    
  # ... other existing config sections
```

## Run States

- **initialized**: Run created, ready to start
- **running**: Currently processing
- **paused**: Temporarily stopped, can be resumed
- **completed**: All processing finished successfully
- **failed**: Run encountered fatal error

## Processing Stages

- **setup**: Initial configuration, hardware initialization, and manager setup
- **operations**: Main observing operations and task execution
- **shutdown**: Hardware shutdown and cleanup

## Output Directory Naming

Directories are automatically named using the pattern:
```
{sanitized_site_name}_{YYYYMMDD}
```

Where YYYYMMDD corresponds to the local midnight date of the observation night.

Examples:
- `Mauna_Kea_Observatory_20250610`
- `Palomar_Mountain_20250615`  
- `Site_34.123_-118.456_20250620` (for unnamed sites)

## Persistence Details

Each run maintains a single JSON state file:

**`run_state.json`**: Human-readable JSON with full run metadata using Pydantic model serialization

The JSON file includes:
- Run identification and timestamps
- Site information and lighting schedule
- Complete command history with timestamps
- List of all generated files
- Error tracking and recovery information
- Configuration snapshot

## Recovery & Error Handling

The system automatically:
- Saves state after every operation
- Tracks errors and retry counts
- Preserves command history for debugging
- Maintains file inventories for cleanup

Failed runs can be inspected, debugged, and resumed from their last successful state.

## Demo

Run the demonstration script to see the system in action:

```bash
python demo_run_management.py
```

This creates a sample run, performs various operations, and demonstrates state persistence and resumability using the integrated configuration system. 