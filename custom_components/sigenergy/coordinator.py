"""DataUpdateCoordinator for Sigenergy Cloud."""
from __future__ import annotations

from datetime import timedelta
import datetime
import logging
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_PASSWORD, CONF_USERNAME
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.storage import Store
from homeassistant.helpers.update_coordinator import (
    DataUpdateCoordinator,
    UpdateFailed,
)
from homeassistant.util import dt as dt_util

from .api import (
    SigenergyApi,
    SigenergyApiError,
    SigenergyAuthError,
    SigenergyRateLimitError,
    SigenergyTransientError,
)
from .const import (
    API_BASE_URL,
    AUTH_METHOD_KEY,
    AUTH_METHOD_PASSWORD,
    CONF_APP_KEY,
    CONF_APP_SECRET,
    CONF_AUTH_METHOD,
    CONF_CACHED_DEVICES,
    CONF_CACHED_SYSTEMS,
    CONF_INSTALLATION_ID,
    CONF_REGION,
    DEFAULT_SCAN_INTERVAL,
    DOMAIN,
    REGION_URLS,
)

_LOGGER = logging.getLogger(__name__)

# Cumulative energy counters (used by TOTAL_INCREASING sensors) mapped to the
# period at which they are legitimately allowed to reset to a lower value.
# `None` means the counter is lifetime and must never decrease.
# The cloud API occasionally returns a corrupted reading (e.g. the value
# divided by 100) for these fields; because TOTAL_INCREASING sensors treat
# any drop as a meter reset, a single bad sample permanently corrupts HA's
# long-term statistics unless it is filtered out before being stored.
_CUMULATIVE_ENERGY_RESET_PERIOD: dict[str, str | None] = {
    "dailyPowerGeneration": "daily",
    "monthlyPowerGeneration": "monthly",
    "annualPowerGeneration": "annual",
    "lifetimePowerGeneration": None,
    "pvEnergyDaily": "daily",
    "pvEnergyTotal": None,
    "esChargingDay": "daily",
    "esDischargingDay": "daily",
    "esDischargingTotal": None,
}

# A drop below this fraction of the previous value is treated as an
# implausible glitch rather than a genuine counter reset.
_GLITCH_DROP_RATIO = 0.5

_STORAGE_VERSION = 1

# 0xFFFFFFFF at the API's decimal scales; Sigenergy uses it to mean "not available".
_INVALID_SENTINELS = frozenset({4294967295.0, 429496729.5, 42949672.95, 4294967.295})


def _is_sentinel(value: Any) -> bool:
    """Return True if `value` is the API's "not available" marker."""
    try:
        return float(value) in _INVALID_SENTINELS
    except (TypeError, ValueError):
        return False


def _strip_sentinels(data: dict[str, Any]) -> dict[str, Any]:
    """Replace "not available" markers with None."""
    return {k: None if _is_sentinel(v) else v for k, v in data.items()}


def _period_changed(
    period: str | None, last: datetime.datetime, now: datetime.datetime
) -> bool:
    """Return True if a reset boundary for `period` lies between `last` and `now`."""
    if period is None:
        return False
    a = dt_util.as_local(last)
    b = dt_util.as_local(now)
    if period == "daily":
        return a.date() != b.date()
    if period == "monthly":
        return (a.year, a.month) != (b.year, b.month)
    if period == "annual":
        return a.year != b.year
    return False


class SigenergyCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Coordinator to manage fetching Sigenergy data from the cloud API."""

    config_entry: ConfigEntry

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(seconds=DEFAULT_SCAN_INTERVAL),
            config_entry=entry,
        )
        self.api = self._create_api(hass, entry)
        self.systems: list[dict[str, Any]] = []
        self.devices: dict[str, list[dict[str, Any]]] = {}
        # Persisted so validation survives HA restarts and data gaps.
        self._store: Store[dict[str, dict[str, Any]]] = Store(
            hass, _STORAGE_VERSION, f"{DOMAIN}.{entry.entry_id}.last_good"
        )
        self._last_good: dict[str, dict[str, Any]] = {}

    def _sanitize_cumulative_values(
        self,
        scope: str,
        new_data: dict[str, Any],
        now: datetime.datetime,
    ) -> dict[str, Any]:
        """Replace implausible drops in cumulative counters with the last known-good value."""
        new_data = _strip_sentinels(new_data)
        for key, period in _CUMULATIVE_ENERGY_RESET_PERIOD.items():
            if key not in new_data:
                continue
            try:
                new_value = float(new_data[key])
            except (TypeError, ValueError):
                continue
            store_key = f"{scope}:{key}"
            last = self._last_good.get(store_key)
            if last is not None and new_value < last["value"] * _GLITCH_DROP_RATIO:
                last_ts = dt_util.parse_datetime(last["ts"])
                if last_ts is not None and not _period_changed(period, last_ts, now):
                    _LOGGER.warning(
                        "Ignoring implausible reading for %s (%s): %s -> %s "
                        "(keeping last known-good value)",
                        key,
                        scope,
                        last["value"],
                        new_value,
                    )
                    new_data[key] = last["value"]
                    continue
            self._last_good[store_key] = {"value": new_value, "ts": now.isoformat()}
        return new_data

    @staticmethod
    def _create_api(
        hass: HomeAssistant, entry: ConfigEntry
    ) -> SigenergyApi:
        """Create API client from config entry."""
        session = async_get_clientsession(hass)
        auth_method = entry.data.get(CONF_AUTH_METHOD, AUTH_METHOD_PASSWORD)

        region = entry.data.get(CONF_REGION)
        base_url = REGION_URLS.get(region, API_BASE_URL)

        if auth_method == AUTH_METHOD_PASSWORD:
            return SigenergyApi(
                session=session,
                auth_method=AUTH_METHOD_PASSWORD,
                username=entry.data[CONF_USERNAME],
                password=entry.data[CONF_PASSWORD],
                base_url=base_url,
            )
        return SigenergyApi(
            session=session,
            auth_method=AUTH_METHOD_KEY,
            app_key=entry.data[CONF_APP_KEY],
            app_secret=entry.data[CONF_APP_SECRET],
            base_url=base_url,
        )

    async def _async_setup(self) -> None:
        """Set up the coordinator: build system list and fetch devices."""
        installation_id = self.config_entry.data.get(CONF_INSTALLATION_ID)
        cached_systems = self.config_entry.data.get(CONF_CACHED_SYSTEMS)
        cached_devices = self.config_entry.data.get(CONF_CACHED_DEVICES, {})
        self._last_good = {
            k: v
            for k, v in (await self._store.async_load() or {}).items()
            if not _is_sentinel(v.get("value"))
        }

        try:
            if installation_id:
                self.systems = [{"systemId": installation_id}]
            elif cached_systems:
                self.systems = cached_systems
                _LOGGER.debug(
                    "Using cached system list (%d system(s)) — skipping API call",
                    len(self.systems),
                )
            else:
                self.systems = await self.api.get_system_list()

            new_cached_devices: dict[str, Any] = dict(cached_devices)
            data_changed = not cached_systems

            for system in self.systems:
                system_id = system["systemId"]
                if system_id in cached_devices:
                    self.devices[system_id] = cached_devices[system_id]
                    _LOGGER.debug(
                        "Using cached device list for system %s", system_id
                    )
                else:
                    self.devices[system_id] = await self.api.get_device_list(system_id)
                    new_cached_devices[system_id] = self.devices[system_id]
                    data_changed = True

            if data_changed:
                self.hass.config_entries.async_update_entry(
                    self.config_entry,
                    data={
                        **self.config_entry.data,
                        CONF_CACHED_SYSTEMS: self.systems,
                        CONF_CACHED_DEVICES: new_cached_devices,
                    },
                )

        except SigenergyRateLimitError as err:
            if cached_systems:
                # Rate limited but we have a cache — use it and continue
                _LOGGER.warning(
                    "Sigenergy API rate limit hit during setup, using cached data: %s", err
                )
                self.systems = cached_systems
                self.devices = {k: v for k, v in cached_devices.items()}
            else:
                raise UpdateFailed(
                    "Sigenergy API rate limit reached (max 1 request/5 min for system list). "
                    "Home Assistant will retry automatically — usually within a few minutes."
                ) from err
        except SigenergyAuthError as err:
            raise ConfigEntryAuthFailed(
                "Authentication failed. Please check your credentials under "
                "Settings → Devices & Services → Sigenergy Cloud → Reconfigure."
            ) from err
        except SigenergyApiError as err:
            raise UpdateFailed(
                f"Could not connect to the Sigenergy API: {err}"
            ) from err

    def _prev_system(self, system_id: str) -> dict[str, Any]:
        """Return previous data for a system, or empty dict."""
        if self.data:
            return self.data.get("systems", {}).get(system_id, {})
        return {}

    async def _async_update_data(self) -> dict[str, Any]:
        """Fetch data from the Sigenergy API.

        On transient errors (rpc fail, station disconnect) the coordinator
        keeps the last-known-good value for the affected section so that
        sensors stay available instead of flipping to 'unavailable'.
        """
        try:
            result: dict[str, Any] = {"systems": {}}
            now = datetime.datetime.now(datetime.timezone.utc)

            for system in self.systems:
                system_id = system["systemId"]
                prev = self._prev_system(system_id)

                system_data: dict[str, Any] = {
                    "info": system,
                    "summary": {},
                    "energy_flow": {},
                    "operating_mode": None,
                    "devices": {},
                }

                # ── System-level realtime data ────────────────────
                try:
                    system_data["summary"] = self._sanitize_cumulative_values(
                        system_id,
                        await self.api.get_realtime_summary(system_id),
                        now,
                    )
                except SigenergyTransientError:
                    system_data["summary"] = prev.get("summary", {})
                except SigenergyApiError as err:
                    _LOGGER.debug("Error fetching summary for %s: %s", system_id, err)
                    system_data["summary"] = prev.get("summary", {})

                try:
                    system_data["energy_flow"] = _strip_sentinels(
                        await self.api.get_energy_flow(system_id)
                    )
                except SigenergyTransientError:
                    system_data["energy_flow"] = prev.get("energy_flow", {})
                except SigenergyApiError as err:
                    _LOGGER.debug(
                        "Error fetching energy flow for %s: %s", system_id, err
                    )
                    system_data["energy_flow"] = prev.get("energy_flow", {})

                # ── Operating mode ────────────────────────────────
                try:
                    system_data["operating_mode"] = await self.api.get_operating_mode(
                        system_id
                    )
                except SigenergyTransientError:
                    system_data["operating_mode"] = prev.get("operating_mode")
                except SigenergyApiError as err:
                    _LOGGER.debug(
                        "Error fetching operating mode for %s: %s", system_id, err
                    )
                    system_data["operating_mode"] = prev.get("operating_mode")

                # ── Device-level realtime data ────────────────────
                prev_devices = prev.get("devices", {})
                devices = self.devices.get(system_id, [])
                for device in devices:
                    serial = device.get("serialNumber", "")
                    try:
                        device_data = await self.api.get_device_realtime(
                            system_id, serial
                        )
                        system_data["devices"][serial] = {
                            "info": device,
                            "realtime": self._sanitize_cumulative_values(
                                f"{system_id}:{serial}",
                                device_data.get("realTimeInfo", {}),
                                now,
                            ),
                        }
                    except SigenergyTransientError:
                        if serial in prev_devices:
                            system_data["devices"][serial] = prev_devices[serial]
                    except SigenergyApiError as err:
                        _LOGGER.debug(
                            "Error fetching device %s data: %s", serial, err
                        )
                        if serial in prev_devices:
                            system_data["devices"][serial] = prev_devices[serial]

                result["systems"][system_id] = system_data

            self._store.async_delay_save(lambda: self._last_good, 60)
            result["last_updated"] = now
            return result

        except SigenergyAuthError as err:
            raise ConfigEntryAuthFailed(
                "Authentication failed. Please check your credentials under "
                "Settings → Devices & Services → Sigenergy Cloud → Reconfigure."
            ) from err
        except SigenergyRateLimitError as err:
            raise UpdateFailed(
                "Sigenergy API rate limit reached. Data will refresh automatically "
                "in the next polling cycle (every 5 minutes)."
            ) from err
        except SigenergyApiError as err:
            raise UpdateFailed(f"Could not connect to the Sigenergy API: {err}") from err
