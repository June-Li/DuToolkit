import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(cur_dir, "../")
sys.path.append(os.path.abspath(root_dir))

import json
import pytest
from WORKFLOW.OTHER.OCR.v0.OCRApiDeploy import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


filename_list = os.listdir(cur_dir + "/data")


@pytest.mark.parametrize("filename", filename_list)
def test_upload_ocr_result(filename, client):
    data = {
        "file": (
            open(cur_dir + "/data/" + filename, "rb"),
            filename,
        ),
        "json": json.dumps({"force_cv": "yes", "table_enhanced": True}),
    }
    response = client.post(
        "/upload_ocr_result", data=data, content_type="multipart/form-data"
    )
    assert response.status_code == 200
