"""Connect to curated public datasets that do not require API keys.

Only fixed, reviewed source URLs are exposed in the app; arbitrary URL fetching is
intentionally not supported in the hosted application.
"""
from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from io import BytesIO
import pandas as pd


WORLD_BANK_INDICATORS = {
    "wb_gdp": ("NY.GDP.MKTP.CD", "GDP (current US$)"),
    "wb_population": ("SP.POP.TOTL", "Population, total"),
    "wb_inflation": ("FP.CPI.TOTL.ZG", "Inflation, consumer prices (annual %)"),
    "wb_unemployment": ("SL.UEM.TOTL.ZS", "Unemployment, total (% of labour force)"),
    "wb_agriculture": ("NV.AGR.TOTL.ZS", "Agriculture, forestry and fishing value added (% of GDP)"),
    "wb_co2": ("EN.ATM.CO2E.PC", "CO₂ emissions (metric tons per capita)"),
    "wb_internet": ("IT.NET.USER.ZS", "Individuals using the Internet (% of population)"),
    "wb_electricity": ("EG.ELC.ACCS.ZS", "Access to electricity (% of population)"),
}

OPEN_DATASETS = {
    "wb_gdp": {
        "name": "GDP (current US$) — World Bank",
        "provider": "World Bank Open Data",
        "category": "Economy",
        "description": "Annual GDP by economy, with country and year identifiers.",
        "access": "Public API · no key",
        "url": "https://api.worldbank.org/v2/indicator/NY.GDP.MKTP.CD",
    },
    "wb_population": {
        "name": "Population — World Bank",
        "provider": "World Bank Open Data",
        "category": "Demographics",
        "description": "Annual total population by economy.",
        "access": "Public API · no key",
        "url": "https://api.worldbank.org/v2/indicator/SP.POP.TOTL",
    },
    "wb_inflation": {
        "name": "Inflation — World Bank",
        "provider": "World Bank Open Data",
        "category": "Economy",
        "description": "Annual consumer price inflation by economy.",
        "access": "Public API · no key",
        "url": "https://api.worldbank.org/v2/indicator/FP.CPI.TOTL.ZG",
    },
    "wb_unemployment": {
        "name": "Unemployment — World Bank",
        "provider": "World Bank Open Data",
        "category": "Labour",
        "description": "ILO modelled unemployment rate by economy.",
        "access": "Public API · no key",
        "url": "https://api.worldbank.org/v2/indicator/SL.UEM.TOTL.ZS",
    },
    "wb_agriculture": {
        "name": "Agriculture value added — World Bank",
        "provider": "World Bank Open Data",
        "category": "Agriculture",
        "description": "Agriculture, forestry and fishing value added as a share of GDP.",
        "access": "Public API · no key",
        "url": "https://api.worldbank.org/v2/indicator/NV.AGR.TOTL.ZS",
    },
    "wb_co2": {
        "name": "CO₂ emissions per capita — World Bank",
        "provider": "World Bank Open Data",
        "category": "Climate",
        "description": "Annual carbon dioxide emissions per person by economy.",
        "access": "Public API · no key",
        "url": "https://api.worldbank.org/v2/indicator/EN.ATM.CO2E.PC",
    },
    "wb_internet": {
        "name": "Internet use — World Bank",
        "provider": "World Bank Open Data",
        "category": "Digital development",
        "description": "Individuals using the Internet as a percentage of population.",
        "access": "Public API · no key",
        "url": "https://api.worldbank.org/v2/indicator/IT.NET.USER.ZS",
    },
    "wb_electricity": {
        "name": "Access to electricity — World Bank",
        "provider": "World Bank Open Data",
        "category": "Energy",
        "description": "Population with access to electricity, by economy.",
        "access": "Public API · no key",
        "url": "https://api.worldbank.org/v2/indicator/EG.ELC.ACCS.ZS",
    },
    "owid_co2": {
        "name": "CO₂ emissions per capita — Our World in Data",
        "provider": "Our World in Data",
        "category": "Climate",
        "description": "Country-level historical CO₂ emissions per capita.",
        "access": "Public CSV · no key",
        "url": "https://ourworldindata.org/grapher/co2-per-capita",
    },
    "owid_life_expectancy": {
        "name": "Life expectancy — Our World in Data",
        "provider": "Our World in Data",
        "category": "Health",
        "description": "Historical life expectancy at birth by country.",
        "access": "Public CSV · no key",
        "url": "https://ourworldindata.org/grapher/life-expectancy",
    },
    "owid_gdp_per_capita": {
        "name": "GDP per capita — Our World in Data",
        "provider": "Our World in Data",
        "category": "Economy",
        "description": "Historical GDP per capita series by country.",
        "access": "Public CSV · no key",
        "url": "https://ourworldindata.org/grapher/gdp-per-capita-worldbank",
    },
    "owid_population": {
        "name": "Population — Our World in Data",
        "provider": "Our World in Data",
        "category": "Demographics",
        "description": "Historical population estimates by country.",
        "access": "Public CSV · no key",
        "url": "https://ourworldindata.org/grapher/population",
    },
}


def list_open_datasets() -> dict:
    """Return a copy of the curated public dataset catalogue."""
    return {key: value.copy() for key, value in OPEN_DATASETS.items()}


def _request_bytes(url: str, timeout: int = 35) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": "AI-Data-Analyst/1.0 (public open-data connector)",
            "Accept": "application/json,text/csv,text/plain,*/*",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"Source returned HTTP {response.status}.")
            payload = response.read(35 * 1024 * 1024 + 1)
    except HTTPError as exc:
        raise RuntimeError(f"Source returned HTTP {exc.code}.") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach the public data source: {exc.reason}") from exc
    if len(payload) > 35 * 1024 * 1024:
        raise RuntimeError("The selected public dataset exceeds the 35 MB download limit.")
    return payload


def _load_world_bank(source_id: str, timeout: int) -> pd.DataFrame:
    indicator, indicator_name = WORLD_BANK_INDICATORS[source_id]
    url = (
        "https://api.worldbank.org/v2/country/all/indicator/"
        f"{quote(indicator, safe='')}?format=json&per_page=20000&date=1960:2026"
    )
    payload = _request_bytes(url, timeout=timeout)
    try:
        response = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("World Bank returned an unreadable response.") from exc
    if not isinstance(response, list) or len(response) < 2 or not isinstance(response[1], list):
        message = response[0].get("message", [{}])[0].get("value", "No observations returned.") if isinstance(response, list) and response else "No observations returned."
        raise RuntimeError(f"World Bank data unavailable: {message}")
    rows = []
    for item in response[1]:
        country = item.get("country") or {}
        rows.append({
            "Country": country.get("value"),
            "Country Code": item.get("countryiso3code"),
            "Year": pd.to_numeric(item.get("date"), errors="coerce"),
            indicator_name: item.get("value"),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("The World Bank API returned no observations for this indicator.")
    frame = frame.dropna(subset=["Country", "Year"])
    frame["Year"] = frame["Year"].astype("Int64")
    frame[indicator_name] = pd.to_numeric(frame[indicator_name], errors="coerce")
    frame = frame.dropna(subset=[indicator_name]).reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("No non-missing observations were available for this indicator.")
    return frame


def load_open_dataset(source_id: str, timeout: int = 35) -> tuple[pd.DataFrame, dict]:
    """Download one curated public dataset without credentials and return data + metadata."""
    if source_id not in OPEN_DATASETS:
        raise ValueError("Choose a dataset from the supported public-source catalogue.")
    metadata = OPEN_DATASETS[source_id].copy()
    if source_id in WORLD_BANK_INDICATORS:
        frame = _load_world_bank(source_id, timeout)
    else:
        url = metadata["url"] + ".csv"
        payload = _request_bytes(url, timeout=timeout)
        try:
            frame = pd.read_csv(BytesIO(payload), low_memory=False)
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            raise RuntimeError("The public CSV could not be parsed as a table.") from exc
        if frame.empty:
            raise RuntimeError("The public source returned an empty dataset.")
    if frame.empty or len(frame.columns) == 0:
        raise RuntimeError("The selected source did not return a usable dataset.")
    metadata.update({
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "loaded_at": pd.Timestamp.now(tz="UTC").isoformat(),
    })
    return frame, metadata
