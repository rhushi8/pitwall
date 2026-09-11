"""Thin async client for the OpenF1 REST API, documented at https://openf1.org

Used for race-day data: stints, pit stops, positions and car data.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional
from urllib.parse import urlencode

import httpx
import pandas as pd

from config.settings import OPENF1_BASE_URL

log = logging.getLogger(__name__)

# Core client

class OpenF1Client:
    """Async HTTP client wrapping the OpenF1 API."""

    def __init__(self, base_url: str = OPENF1_BASE_URL, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def get(self, endpoint: str, **params: Any) -> list[dict]:
        """GET /v1/{endpoint}?key=val&... → list of result dicts."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        if params:
            url += "?" + urlencode(
                {k: v for k, v in params.items() if v is not None}
            )
        log.debug("GET %s", url)
        try:
            client = await self._get_client()
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
        except httpx.TimeoutException:
            log.error("Timeout fetching endpoint '%s'", endpoint)
            return []
        except httpx.HTTPStatusError as exc:
            log.error("HTTP %s fetching endpoint '%s': %s", exc.response.status_code, endpoint, exc)
            return []
        except httpx.HTTPError as exc:
            log.error("HTTP error fetching endpoint '%s': %s", endpoint, exc)
            return []
        except Exception as exc:
            log.exception("Unexpected OpenF1 error on endpoint '%s': %s", endpoint, exc)
            return []

    # Synchronous convenience wrapper
    def get_sync(self, endpoint: str, **params: Any) -> list[dict]:
        try:
            return asyncio.run(self.get(endpoint, **params))
        except RuntimeError:
            # Fallback for environments with an already-running loop.
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self.get(endpoint, **params))
            finally:
                loop.run_until_complete(self.close())
                loop.close()


# Domain helpers (each returns a cleaned DataFrame)

class F1DataFetcher:
    """High-level fetcher for OpenF1 race-weekend data."""

    def __init__(self, client: Optional[OpenF1Client] = None):
        self.client = client or OpenF1Client()

    # Sessions

    def get_sessions(self, year: int, gp_name: Optional[str] = None) -> pd.DataFrame:
        """List all sessions in a season (or filter by GP name)."""
        params: dict = {"year": year}
        if gp_name:
            params["circuit_short_name"] = gp_name
        data = self.client.get_sync("sessions", **params)
        return pd.DataFrame(data)

    def get_session_key(self, year: int, gp_name: str, session_type: str) -> Optional[int]:
        """Resolve a (year, gp, session_type) tuple to an OpenF1 session_key."""
        df = self.get_sessions(year, gp_name)
        if df.empty:
            return None
        match = df[df["session_type"].str.upper() == session_type.upper()]
        if match.empty:
            return None
        return int(match.iloc[0]["session_key"])

    # Stints

    def get_stints(self, session_key: int) -> pd.DataFrame:
        """
        Returns all stints for a session.

        Key columns: driver_number, stint_number, lap_start, lap_end,
                     compound, tyre_age_at_start
        """
        data = self.client.get_sync("stints", session_key=session_key)
        df = pd.DataFrame(data)
        if df.empty:
            return df
        df["lap_count"] = df["lap_end"] - df["lap_start"] + 1
        return df

    # Pit stops

    def get_pit_stops(self, session_key: int) -> pd.DataFrame:
        """
        Returns all pit stops for a session.
        Key column: pit_duration (seconds)
        """
        data = self.client.get_sync("pit", session_key=session_key)
        return pd.DataFrame(data)

    def get_team_pit_averages(self, session_key: int) -> pd.DataFrame:
        """Returns mean pit stop time per team for a given session."""
        pit = self.get_pit_stops(session_key)
        if pit.empty or "pit_duration" not in pit.columns:
            return pd.DataFrame()
        if "team_name" not in pit.columns:
            return pd.DataFrame()
        return (
            pit.groupby("team_name")["pit_duration"]
            .agg(mean="mean", std="std", count="count")
            .reset_index()
            .rename(columns={"mean": "pit_time_mean_s", "std": "pit_time_std_s"})
        )

    # Lap times

    def get_laps(self, session_key: int,
                 driver_number: Optional[int] = None) -> pd.DataFrame:
        """
        Returns per-lap data for a session (optionally filtered by driver).
        Key columns: driver_number, lap_number, lap_duration,
                     duration_sector_1/2/3, is_pit_out_lap
        """
        params: dict = {"session_key": session_key}
        if driver_number:
            params["driver_number"] = driver_number
        data = self.client.get_sync("laps", **params)
        return pd.DataFrame(data)

    # Car data (position stream)

    def get_positions(self, session_key: int) -> pd.DataFrame:
        """
        Returns position-update stream for a session.
        Useful for race-progress simulation and safety-car period detection.
        """
        data = self.client.get_sync("position", session_key=session_key)
        return pd.DataFrame(data)

    # Weather

    def get_weather(self, session_key: int) -> pd.DataFrame:
        """Returns time-series weather data for a session."""
        data = self.client.get_sync("weather", session_key=session_key)
        return pd.DataFrame(data)

    def get_race_day_weather_summary(self, session_key: int) -> dict:
        """Summarises race-day weather into scalar features."""
        df = self.get_weather(session_key)
        if df.empty:
            return {}
        return {
            "race_air_temp_mean":   round(df["air_temperature"].mean(), 1) if "air_temperature" in df else None,
            "race_track_temp_mean": round(df["track_temperature"].mean(), 1) if "track_temperature" in df else None,
            "race_humidity_mean":   round(df["humidity"].mean(), 1) if "humidity" in df else None,
            "race_rainfall":        bool(df["rainfall"].any()) if "rainfall" in df else False,
        }

    # Drivers

    def get_drivers(self, session_key: int) -> pd.DataFrame:
        """Returns driver metadata (number, code, team) for a session."""
        data = self.client.get_sync("drivers", session_key=session_key)
        return pd.DataFrame(data)

    # Compound usage summary

    def get_compound_strategy_summary(self, session_key: int) -> pd.DataFrame:
        """
        Returns a per-driver stint strategy summary for a race session.

        Columns: driver_number, n_stints, compounds_used, first_compound,
                 total_pit_stops
        """
        stints = self.get_stints(session_key)
        if stints.empty:
            log.warning("No stints found for session_key=%s", session_key)
            return pd.DataFrame()

        def summarise(g: pd.DataFrame) -> pd.Series:
            return pd.Series(
                {
                    "n_stints":       len(g),
                    "compounds_used": list(g["compound"].unique()),
                    "first_compound": g.sort_values("stint_number").iloc[0]["compound"],
                    "total_pit_stops":len(g) - 1,
                }
            )

        return (
            stints.groupby("driver_number")
            .apply(summarise)
            .reset_index()
        )


# Utility: merge OpenF1 live data with a FastF1 feature DataFrame

def enrich_with_openf1(
    feature_df: pd.DataFrame,
    year: int,
    gp_name: str,
    fetcher: Optional[F1DataFetcher] = None,
) -> pd.DataFrame:
    """
    Adds OpenF1-sourced columns (pit times, strategy, live weather)
    to an existing per-driver feature DataFrame produced by FastF1.

    Merges on driver_code (3-letter abbreviation).
    """
    fetcher = fetcher or F1DataFetcher()
    session_key = fetcher.get_session_key(year, gp_name, "R")
    if session_key is None:
        log.warning("No race session_key found for %s %s, skipping enrichment", year, gp_name)
        return feature_df

    drivers = fetcher.get_drivers(session_key)
    strategy = fetcher.get_compound_strategy_summary(session_key)
    pit_avgs = fetcher.get_team_pit_averages(session_key)
    weather  = fetcher.get_race_day_weather_summary(session_key)

    # Map driver_number → driver_code
    if not drivers.empty and "name_acronym" in drivers.columns:
        num_to_code = drivers.set_index("driver_number")["name_acronym"].to_dict()
        if not strategy.empty:
            strategy["driver_code"] = strategy["driver_number"].map(num_to_code)
            feature_df = feature_df.merge(
                strategy.drop(columns=["driver_number"], errors="ignore"),
                on="driver_code",
                how="left",
            )

    # Merge team pit averages
    if not pit_avgs.empty and "team_name" in feature_df.columns:
        feature_df = feature_df.merge(
            pit_avgs[["team_name", "pit_time_mean_s"]],
            on="team_name",
            how="left",
        )

    # Add race-day weather scalars
    for k, v in weather.items():
        feature_df[k] = v

    return feature_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s - %(message)s")
    fetcher = F1DataFetcher()
    sessions = fetcher.get_sessions(2024)
    print(sessions[["session_key", "session_name", "date_start"]].head(10))
