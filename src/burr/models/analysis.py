from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


class TimestampedValue(BaseModel):
    """A value with an associated timestamp"""

    value: Any
    timestamp: datetime


class AnalysisResults(BaseModel):
    """Container for analysis results with timestamps"""

    # Store values with their timestamps
    values: Dict[str, TimestampedValue] = Field(default_factory=dict)

    def set_value(self, key: str, value: Any, timestamp: Optional[datetime] = None):
        """Set a value with an optional timestamp"""
        if timestamp is None:
            timestamp = datetime.now(ZoneInfo("UTC"))

        self.values[key] = TimestampedValue(value=value, timestamp=timestamp)

    def get_latest_value(self, key: str, max_age: Optional[timedelta] = None, default: Any = None) -> Any:
        """
        Get the latest value for a key, optionally checking age

        Parameters
        ----------
        key : str
            The key to look up
        max_age : timedelta, optional
            Maximum age of the value to accept
        default : Any, optional
            Default value if no valid value found

        Returns
        -------
        Any
            The value if found and valid, otherwise default
        """
        if key not in self.values:
            return default

        timestamped = self.values[key]

        if max_age is not None:
            age = datetime.now(ZoneInfo("UTC")) - timestamped.timestamp
            if age > max_age:
                return default

        return timestamped.value

    def get_value_timestamp(self, key: str) -> Optional[datetime]:
        """Get the timestamp for a value"""
        if key not in self.values:
            return None
        return self.values[key].timestamp
