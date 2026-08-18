"""
API tests.

Each test gets a throwaway SQLite database, so they can run in any order and
in parallel without stepping on each other.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from app.db.models import Base, engine
from app.main import app


@pytest.fixture(autouse=True)
def fresh_database():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def csv_upload() -> tuple[str, io.BytesIO, str]:
    rows = ["x,y,group"]
    rows += [f"{i},{i * 2 + (i % 7)},{'AB'[i % 2]}" for i in range(80)]
    content = "\n".join(rows).encode()
    return ("data.csv", io.BytesIO(content), "text/csv")


class TestHealth:
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_root_lists_entrypoints(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "docs" in response.json()


class TestUpload:
    def test_accepts_a_csv(self, client, csv_upload):
        response = client.post("/api/analyses", files={"file": csv_upload})

        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "running"
        assert body["rows"] == 80
        assert body["analysis_id"]

    def test_rejects_a_text_file(self, client):
        response = client.post(
            "/api/analyses",
            files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert response.status_code == 400
        assert "supported format" in response.json()["detail"]

    def test_rejects_a_single_column_csv(self, client):
        response = client.post(
            "/api/analyses",
            files={"file": ("thin.csv", io.BytesIO(b"a\n1\n2\n"), "text/csv")},
        )
        assert response.status_code == 400


class TestRetrieval:
    def test_returns_the_completed_analysis(self, client, csv_upload):
        # TestClient runs background tasks synchronously, so it's done on return
        analysis_id = client.post("/api/analyses", files={"file": csv_upload}).json()["analysis_id"]

        response = client.get(f"/api/analyses/{analysis_id}")
        assert response.status_code == 200

        body = response.json()
        assert body["status"] == "completed"
        assert body["report"]["markdown"]
        assert len(body["charts"]) > 0

    def test_returns_the_report_as_markdown(self, client, csv_upload):
        analysis_id = client.post("/api/analyses", files={"file": csv_upload}).json()["analysis_id"]

        response = client.get(f"/api/analyses/{analysis_id}/report")
        assert response.status_code == 200
        assert response.text.startswith("# Analysis:")

    def test_unknown_id_returns_404(self, client):
        response = client.get("/api/analyses/does-not-exist")
        assert response.status_code == 404

    def test_lists_analyses(self, client, csv_upload):
        client.post("/api/analyses", files={"file": csv_upload})

        response = client.get("/api/analyses")
        assert response.status_code == 200
        assert len(response.json()["analyses"]) == 1

    def test_deletes_an_analysis(self, client, csv_upload):
        analysis_id = client.post("/api/analyses", files={"file": csv_upload}).json()["analysis_id"]

        assert client.delete(f"/api/analyses/{analysis_id}").status_code == 204
        assert client.get(f"/api/analyses/{analysis_id}").status_code == 404


class TestSerialization:
    """Regression tests for a real bug: NaN and Timestamps leaking into JSON.

    `json.dumps` rejects both, so an upload with missing values or a date
    column used to return a 500 once the results were fetched.
    """

    def test_handles_missing_values_and_dates(self, client):
        rows = ["value,other,when"]
        for i in range(60):
            value = "" if i % 10 == 0 else str(i)  # deliberate gaps
            rows.append(f"{value},{i * 3},2024-01-{i % 28 + 1:02d}")
        upload = ("gappy.csv", io.BytesIO("\n".join(rows).encode()), "text/csv")

        analysis_id = client.post("/api/analyses", files={"file": upload}).json()[
            "analysis_id"
        ]

        response = client.get(f"/api/analyses/{analysis_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "completed"

    def test_stored_payload_is_json_encodable(self, client, csv_upload):
        import json

        analysis_id = client.post("/api/analyses", files={"file": csv_upload}).json()[
            "analysis_id"
        ]
        body = client.get(f"/api/analyses/{analysis_id}").json()

        # allow_nan=False is what a strict JSON encoder does — this is the
        # check that would have caught the original bug
        json.dumps(body, allow_nan=False)
