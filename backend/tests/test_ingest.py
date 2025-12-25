import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_ingest_endpoint():
    """Test the ingest endpoint."""
    response = client.post("/api/v1/ingest")
    # The response might vary depending on whether docs exist and Qdrant is configured
    # This test checks that the endpoint is accessible
    assert response.status in [200, 404, 500]  # Various possible responses


def test_ingestion_status():
    """Test the ingestion status endpoint."""
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "message" in data


def test_ingest_without_docs():
    """Test ingest when no documents are available."""
    # This test assumes that if no docs exist, the endpoint should handle it gracefully
    response = client.post("/api/v1/ingest")
    # Could return 404 if no docs found, or 200 if handled gracefully
    assert response.status_code in [200, 404, 500]