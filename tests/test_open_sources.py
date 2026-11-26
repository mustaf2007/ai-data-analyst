import json

import pandas as pd
import pytest

import open_sources


def test_catalog_contains_no_key_sources_and_known_providers():
    catalog = open_sources.list_open_datasets()
    assert "wb_gdp" in catalog
    assert "owid_co2" in catalog
    assert all("no key" in item["access"].lower() for item in catalog.values())


def test_load_our_world_in_data_csv_without_api_key(monkeypatch):
    csv_payload = b"Entity,Code,Year,CO2 per capita\nSomalia,SOM,2020,0.12\nKenya,KEN,2020,0.35\n"
    monkeypatch.setattr(open_sources, "_request_bytes", lambda url, timeout=35: csv_payload)

    frame, metadata = open_sources.load_open_dataset("owid_co2")

    assert len(frame) == 2
    assert frame.loc[0, "Entity"] == "Somalia"
    assert metadata["provider"] == "Our World in Data"
    assert metadata["rows"] == 2
    assert metadata["loaded_at"].endswith("+00:00")


def test_load_world_bank_indicator_flattens_country_year_observations(monkeypatch):
    payload = [
        {"page": 1, "pages": 1},
        [
            {
                "country": {"value": "Somalia"},
                "countryiso3code": "SOM",
                "date": "2022",
                "value": 10.5,
            },
            {
                "country": {"value": "Kenya"},
                "countryiso3code": "KEN",
                "date": "2022",
                "value": 20.5,
            },
        ],
    ]
    monkeypatch.setattr(
        open_sources,
        "_request_bytes",
        lambda url, timeout=35: json.dumps(payload).encode("utf-8"),
    )

    frame, metadata = open_sources.load_open_dataset("wb_gdp")

    assert list(frame.columns) == [
        "Country", "Country Code", "Year", "GDP (current US$)"
    ]
    assert frame["Country"].tolist() == ["Somalia", "Kenya"]
    assert frame["Year"].astype(int).tolist() == [2022, 2022]
    assert metadata["provider"] == "World Bank Open Data"


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError, match="supported public-source catalogue"):
        open_sources.load_open_dataset("arbitrary_url")


def test_download_size_limit_is_enforced(monkeypatch):
    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size):
            return b"x" * size

    monkeypatch.setattr(open_sources, "urlopen", lambda *args, **kwargs: FakeResponse())
    with pytest.raises(RuntimeError, match="35 MB download limit"):
        open_sources._request_bytes("https://example.org/data.csv")
