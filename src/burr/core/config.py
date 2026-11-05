import logging
from pathlib import Path
from typing import Literal, Optional, List
import yaml
from pydantic import BaseModel, ConfigDict, Field

from burr.models.site import SiteMetadata
from burr.models.hardware import HardwareConfig


logger = logging.getLogger(__name__)

# Global singleton instance
_config_instance: Optional["AppConfig"] = None


def load_yaml(path: Path) -> dict:
    """Load YAML file into dictionary (returns contents of the ``app`` key)."""
    try:
        with path.open() as f:
            data = yaml.safe_load(f)
            return data.get("app", {})
    except Exception as e:
        logger.error(f"Failed to load config from {path}: {e}")
        return {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` on top of ``base``. Returns a new dict.

    Scalar and list values in ``override`` replace those in ``base``.
    Nested dicts are merged key-by-key.
    """
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_yaml_merged(paths: list[Path]) -> dict:
    """Load multiple YAML config files and deep-merge in order.

    Later files override earlier ones. Use to compose a site file with a
    preset (mode) file: ``load_yaml_merged([site, preset])`` — preset
    overrides site on any conflicts.
    """
    merged: dict = {}
    for p in paths:
        merged = _deep_merge(merged, load_yaml(p))
    return merged


class LoggingConfig(BaseModel):
    """Logging configuration"""

    level: str = "INFO"

    model_config = ConfigDict(frozen=True)


class NotificationConfig(BaseModel):
    """Notification configuration"""

    slack_channel: Optional[str] = None
    model_config = ConfigDict(frozen=True)


class PlottingConfig(BaseModel):
    """Plotting configuration"""

    debug: bool = Field(description="Debug Plots")
    review: bool = Field(description="Review Plots")
    generate_plots: bool = Field(default=True, description="Generate plots")
    plot_format: str = Field(default="png", description="Plot format (png, pdf, svg)")
    plot_dpi: int = Field(default=150, description="Plot DPI", ge=50)


class RuntimeConfig(BaseModel):
    """CLI runtime configuration"""

    run_id: str = Field(default="burr", description="Run identifier")
    output_dir: str = Field(default=".", description="Output directory")

    model_config = ConfigDict(frozen=False)  # Allow updates to Runtime config


class SchedulingConfig(BaseModel):
    """Base scheduling configuration."""

    type: str = Field(description="Type of scheduling: time_share, interval, or one_shot")

    model_config = ConfigDict(frozen=True)


class TimeShareSchedulingConfig(SchedulingConfig):
    """Configuration for time-share tasks."""

    type: str = "time_share"
    target_percentage: float = Field(description="Target percentage of time to allocate", ge=0, le=100)


class IntervalSchedulingConfig(SchedulingConfig):
    """Configuration for interval-based tasks."""

    type: str = "interval"
    minimum_interval_minutes: int = Field(description="Minimum minutes between executions", ge=1)
    fill_gaps: bool = Field(default=True, description="Whether to use this task to fill gaps")


class OneTimeSchedulingConfig(SchedulingConfig):
    """Configuration for one-time tasks."""

    type: str = "one_shot"


class TwilightFlatsConfig(BaseModel):
    """Configuration for twilight flat collection."""

    windows: List[str] = Field(default_factory=list, description="Lighting conditions for collection")
    collect: bool = Field(default=True, description="Whether to collect flats")
    n_frames: int = Field(default=20, description="Number of frames to collect")
    minimum_exposure_seconds: float = Field(default=0.1, description="Minimum exposure time in seconds")
    min_counts: int = Field(default=30000, description="Minimum ADU counts")
    max_counts: int = Field(default=50000, description="Maximum ADU counts")
    max_saturation_percentage: float = Field(default=1.0, description="Maximum percentage of saturated pixels allowed")
    saturation_threshold: Optional[int] = Field(
        default=None, description="Saturation threshold in ADU (auto-detect if None)"
    )
    long_exposure_threshold: float = Field(
        default=5.0, description="Deprecated: superseded by sky_doubling_seconds; accepted and ignored"
    )
    maximum_exposure_seconds: float = Field(default=60.0, description="Maximum exposure time in seconds")
    sky_doubling_seconds: float = Field(
        default=180.0,
        description=(
            "Twilight sky brightness changes by 2x every this many seconds. The next exposure is "
            "scaled by 2**(elapsed/sky_doubling_seconds): longer at dusk, shorter at dawn"
        ),
    )
    cutout_size: Optional[int] = Field(
        default=300,
        description="Side of the central box (pixels) used for count statistics; limits vignetting bias. None = full frame",
    )
    frame_settle_seconds: float = Field(
        default=30.0, description="SK mode: how long to wait for camera FITS writes to land after a collect"
    )
    scheduling: OneTimeSchedulingConfig = Field(
        default_factory=OneTimeSchedulingConfig, description="Scheduling configuration"
    )

    model_config = ConfigDict(frozen=True)


class DarksConfig(BaseModel):
    """Configuration for dark frame collection."""

    windows: List[str] = Field(default_factory=list, description="Lighting conditions for collection")
    collect: bool = Field(default=True, description="Whether to collect dark frames")
    exposures_per_iteration: int = Field(default=10, description="Number of exposures per iteration")
    scheduling: OneTimeSchedulingConfig = Field(
        default_factory=OneTimeSchedulingConfig, description="Scheduling configuration"
    )


class CalsatsConfig(BaseModel):
    """Configuration for calibration satellite observations."""

    windows: List[str] = Field(default_factory=list, description="Lighting conditions for collection")
    collect: bool = Field(default=True, description="Whether to collect calsat observations")
    minimum_exposure_seconds: float = Field(default=1.0, description="Minimum exposure time")
    maximum_exposure_seconds: float = Field(default=10.0, description="Maximum exposure time")
    minimum_altitude_degrees: float = Field(default=40.0, description="Minimum satellite altitude")
    maximum_altitude_degrees: float = Field(default=85.0, description="Maximum satellite altitude")
    tle_provider: str = Field(
        default="spacetrack",
        description="TLE source: 'spacetrack' (requires login) or 'spacebook' (free, COMSPOC)",
    )
    scheduling: IntervalSchedulingConfig = Field(
        default_factory=lambda: IntervalSchedulingConfig(minimum_interval_minutes=30, fill_gaps=True),
        description="Scheduling configuration",
    )

    model_config = ConfigDict(frozen=True)


class PhotometricStandardsConfig(BaseModel):
    """Configuration for photometric standard observations."""

    windows: List[str] = Field(default_factory=list, description="Lighting conditions for collection")
    collect: bool = Field(default=True, description="Whether to collect standard observations")
    minimum_exposure_seconds: float = Field(default=1.0, description="Minimum exposure time")
    maximum_exposure_seconds: float = Field(default=10.0, description="Maximum exposure time")
    n_exposures: int = Field(default=10, description="Number of exposures per standard")
    minimum_altitude_degrees: float = Field(default=40.0, description="Minimum standard altitude")
    maximum_altitude_degrees: float = Field(default=85.0, description="Maximum standard altitude")
    scheduling: TimeShareSchedulingConfig = Field(
        default_factory=lambda: TimeShareSchedulingConfig(target_percentage=50), description="Scheduling configuration"
    )

    model_config = ConfigDict(frozen=True)


class CoverageConfig(BaseModel):
    """Configuration for sky coverage observations."""

    windows: List[str] = Field(default_factory=list, description="Lighting conditions for collection")
    collect: bool = Field(default=True, description="Whether to collect coverage observations")
    minimum_altitude_degrees: float = Field(default=40.0, description="Minimum altitude for observations in degrees")
    maximum_altitude_degrees: float = Field(default=85.0, description="Maximum altitude for observations in degrees")
    minimum_exposure_seconds: float = Field(default=1.0, description="Minimum exposure time")
    maximum_exposure_seconds: float = Field(default=10.0, description="Maximum exposure time")
    n_exposures: int = Field(default=3, description="Number of exposures per pointing")
    points_per_map: int = Field(default=12, description="Number of points per coverage map")
    min_separation_degrees: float = Field(default=10.0, description="Minimum separation between points")
    enable_plotting: bool = Field(default=True, description="Whether to generate coverage plots")
    scheduling: TimeShareSchedulingConfig = Field(
        default_factory=lambda: TimeShareSchedulingConfig(target_percentage=30), description="Scheduling configuration"
    )

    model_config = ConfigDict(frozen=True)


class LunarBackgroundConfig(BaseModel):
    """Configuration for lunar background measurements."""

    windows: List[str] = Field(default_factory=list, description="Lighting conditions for collection")
    collect: bool = Field(default=True, description="Whether to collect lunar background measurements")
    minimum_moon_altitude_degrees: float = Field(default=30.0, description="Minimum moon altitude for measurements")
    maximum_moon_altitude_degrees: float = Field(default=85.0, description="Maximum moon altitude for measurements")
    minimum_moon_phase: float = Field(default=0.2, description="Minimum moon phase (0=new, 1=full)")
    moon_separation_degrees: List[float] = Field(
        default_factory=lambda: [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
        description="Angular separations from moon to measure at",
    )
    minimum_exposure_seconds: float = Field(default=1.0, description="Minimum exposure time")
    maximum_exposure_seconds: float = Field(default=10.0, description="Maximum exposure time")
    n_exposures: int = Field(default=5, description="Number of exposures per separation")
    scheduling: OneTimeSchedulingConfig = Field(
        default_factory=OneTimeSchedulingConfig, description="Scheduling configuration"
    )

    model_config = ConfigDict(frozen=True)


class StreaksConfig(BaseModel):
    """Configuration for streak observations."""

    collect: bool = Field(default=True, description="Whether to collect streak observations")
    streaks_per_collect: int = Field(default=2, description="Number of streaks to collect per observation")
    minimum_length_pixel: float = Field(default=5.0, description="Minimum streak length in pixels")
    maximum_length_pixel: float = Field(default=100.0, description="Maximum streak length in pixels")
    minimum_orientation_deg: float = Field(default=0.0, description="Minimum streak orientation in degrees")
    maximum_orientation_deg: float = Field(default=180.0, description="Maximum streak orientation in degrees")

    model_config = ConfigDict(frozen=True)


class ScheduleConfig(BaseModel):
    """Configuration for observation scheduling."""

    twilight_flats: TwilightFlatsConfig = Field(
        default_factory=TwilightFlatsConfig, description="Twilight flats collection configuration"
    )
    darks: DarksConfig = Field(default_factory=DarksConfig, description="Dark frame collection configuration")
    calsats: CalsatsConfig = Field(
        default_factory=CalsatsConfig, description="Calibration satellite observation configuration"
    )
    photometric_standards: PhotometricStandardsConfig = Field(
        default_factory=PhotometricStandardsConfig, description="Photometric standards observation configuration"
    )
    coverage: CoverageConfig = Field(
        default_factory=CoverageConfig, description="Sky coverage observation configuration"
    )
    streaks: StreaksConfig = Field(default_factory=StreaksConfig, description="Streak observation configuration")
    lunar_background: LunarBackgroundConfig = Field(
        default_factory=LunarBackgroundConfig, description="Lunar background measurement configuration"
    )

    model_config = ConfigDict(frozen=True)


class NotificationsConfig(BaseModel):
    """Configuration for notifications."""

    enabled: bool = Field(default=True, description="Whether to enable notifications")
    slack_channel: str = Field(default="#burr", description="Slack channel to send notifications to")


class WeatherConfig(BaseModel):
    """Configuration for weather monitoring."""

    # Rain rate monitoring
    rain_rate_threshold: float = Field(
        default=0.0, description="Rain rate threshold in mm/hr - dome closes if exceeded"
    )

    # Humidity monitoring
    humidity_threshold: float = Field(default=85.0, description="Humidity threshold in % - dome closes if exceeded")
    humidity_duration_minutes: int = Field(
        default=5, description="Minutes humidity must be above threshold before closing"
    )

    # Recovery conditions
    recovery_duration_minutes: int = Field(
        default=30, description="Minutes of good weather required before reopening dome"
    )

    # Monitoring frequency
    check_interval_seconds: int = Field(default=60, description="How often to check weather conditions in seconds")

    # Error handling
    max_consecutive_errors: int = Field(
        default=3, description="Maximum consecutive errors before treating as bad weather"
    )

    model_config = ConfigDict(frozen=True)


class SKConfig(BaseModel):
    """Configuration for SensorKit integration.

    Entity names map the logical burr devices/controllers/programs to the
    names registered in the SensorKit backend.

    Note: the burr program entity has no separate ``_entity`` field — its
    name is inherited from the service name (set by the launcher via
    ``sensorkit service run <name> ...`` or the ``name:`` field in
    services.yaml). See ``burr.hardware.sk.burr_program.main``.
    """

    mount_entity: str = Field(default="mymount")
    camera_entity: str = Field(default="mycamera")
    dome_entity: Optional[str] = Field(default="mydome")
    weather_entity: Optional[str] = Field(default="myweather")
    burr_controller_entity: str = Field(default="burr_controller")
    dome_controller_entity: Optional[str] = Field(default="dome_controller")
    data_root: Optional[Path] = Field(
        default=None,
        description=(
            "Directory where the SK camera/data graph writes FITS frames. "
            "Required for task sources that need to read back their own "
            "frames between requests (e.g. twilight flats, which adapt "
            "exposure time from the previous frame's median counts). "
            "Should match the WriteFile node's ``directory`` in the SK "
            "camera service config."
        ),
    )
    rate_target_frame: Literal["icrf", "cirf"] = Field(
        default="icrf",
        description=(
            "Reference frame to label RateTarget objects with when emitting "
            "rate-tracked SK tasks. Mount drivers differ on what they "
            "accept: pwi4 accepts (RateTarget, ICRF), node_platform accepts "
            "(RateTarget, CIRF). Setting the wrong one sends the target "
            "through sensorkit's BaseTarget.adapt() → to_ephemeris_target() "
            "path, which is unimplemented for RateTarget and crashes "
            "FollowTarget with NotImplementedError. Default 'icrf' matches "
            "pwi4; set 'cirf' on Node-Platform sites."
        ),
    )

    model_config = ConfigDict(frozen=True)


class AppConfig(BaseModel):
    """Application configuration"""

    debug: bool = Field(default=False, description="Debug mode")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")
    stage: str = Field(description="Deployment stage")
    plotting: PlottingConfig = Field(default_factory=PlottingConfig, description="Plotting configuration")
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig, description="Runtime configuration options")
    site: SiteMetadata = Field(default_factory=SiteMetadata, description="Site configuration")
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig, description="Schedule configuration")
    notifications: NotificationsConfig = Field(
        default_factory=NotificationsConfig, description="Notifications configuration"
    )
    weather: WeatherConfig = Field(default_factory=WeatherConfig, description="Weather monitoring configuration")
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    sk: SKConfig = Field(default_factory=SKConfig, description="SensorKit integration configuration")

    model_config = ConfigDict(frozen=True)


def get_config() -> AppConfig:
    """Get the global config instance.

    Returns:
        AppConfig: Configuration instance

    Raises:
        RuntimeError: If config hasn't been initialized
    """
    if _config_instance is None:
        raise RuntimeError("Configuration not initialized. Call initialize_config() first.")
    return _config_instance


def initialize_config(config_path: Path | list[Path]) -> AppConfig:
    """Initialize the global config instance.

    Args:
        config_path: Either a single YAML path (legacy single-file config)
            or a list of paths that are deep-merged in order (e.g. a
            site file plus a mode preset). When a list is given, later
            files override earlier ones.

    Returns:
        AppConfig: Configuration instance
    """
    global _config_instance

    paths = config_path if isinstance(config_path, list) else [config_path]

    if len(paths) == 1:
        logger.info(f"Loading configuration from {paths[0]}")
    else:
        logger.info(f"Loading merged configuration from {[str(p) for p in paths]}")

    config_data = load_yaml_merged(paths)

    try:
        config = AppConfig(**config_data)
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise

    _config_instance = config
    return config


def get_or_initialize_config(
    config_path: Path | list[Path] | None = None,
) -> AppConfig:
    """Get a loaded config, if none, load config_path or LOCAL_OVERRIDE.

    Args:
        config_path: Optional path (or list of paths to deep-merge) for
            the config YAML. When a list is given, later files override
            earlier ones — use for composing a site file with a preset.

    Returns:
        AppConfig: Configuration instance
    """
    try:
        config = get_config()
    except RuntimeError:
        if config_path:
            config = initialize_config(config_path)
        else:
            from burr.core.constants import LOCAL_APP_CONFIG_OVERRIDE

            logger.info(f"No config initialized, using {LOCAL_APP_CONFIG_OVERRIDE}")
            config = initialize_config(LOCAL_APP_CONFIG_OVERRIDE)

    return config
