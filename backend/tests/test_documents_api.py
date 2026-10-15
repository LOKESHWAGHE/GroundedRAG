from pathlib import Path

from reportlab.pdfgen import canvas

from app.core.config import get_settings

BASE = "/api/v1/documents"


def upload(client, name, data, content_type="application/octet-stream"):
    return client.post(f"{BASE}/upload", files={"file": (name, data, content_type)})


def test_upload_txt_is_parsed(client):
    r = upload(client, "notes.txt", b"Product X supports MySQL and PostgreSQL.")
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "chunking"  # parsing + chunking run inline on upload; embedding is a separate step
    assert body["page_count"] == 1
    assert body["char_count"] > 0
    assert "stored_path" not in body


def test_pages_endpoint_returns_cleaned_text(client):
    doc = upload(client, "a.txt", b"inter-\nnational   trade").json()
    pages = client.get(f"{BASE}/{doc['id']}/pages").json()
    assert pages[0]["page_number"] == 1
    assert pages[0]["text"] == "international trade"


def test_chunks_endpoint_returns_chunks(client):
    doc = upload(client, "a.txt", b"Product X supports PostgreSQL and MySQL databases.").json()
    chunks = client.get(f"{BASE}/{doc['id']}/chunks").json()
    assert len(chunks) >= 1
    assert "PostgreSQL" in chunks[0]["text"]


def test_multi_page_pdf(client, tmp_path):
    pdf = tmp_path / "x.pdf"
    c = canvas.Canvas(str(pdf))
    c.drawString(72, 720, "alpha")
    c.showPage()
    c.drawString(72, 720, "beta")
    c.showPage()
    c.save()
    body = upload(client, "x.pdf", pdf.read_bytes(), "application/pdf").json()
    assert body["page_count"] == 2


def test_image_only_pdf_is_marked_failed(client, tmp_path):
    pdf = tmp_path / "blank.pdf"
    c = canvas.Canvas(str(pdf))
    c.showPage()
    c.save()
    body = upload(client, "blank.pdf", pdf.read_bytes()).json()
    assert body["status"] == "failed"
    assert "No extractable text" in body["error_message"]


def test_duplicate_upload_returns_409_with_existing_id(client):
    first = upload(client, "a.txt", b"same content").json()
    r = upload(client, "renamed.txt", b"same content")
    assert r.status_code == 409
    assert r.json()["detail"]["existing_document_id"] == first["id"]


def test_unsupported_extension_415(client):
    assert upload(client, "run.exe", b"MZ...").status_code == 415


def test_content_extension_mismatch_400(client):
    assert upload(client, "fake.pdf", b"just text").status_code == 400


def test_empty_file_400(client):
    assert upload(client, "e.txt", b"").status_code == 400


def test_oversize_413(client, monkeypatch):
    monkeypatch.setattr(get_settings(), "MAX_UPLOAD_SIZE_MB", 0)
    assert upload(client, "a.txt", b"x").status_code == 413


def test_path_traversal_filename_is_neutralised(client):
    body = upload(client, "../../evil.txt", b"hello").json()
    assert body["filename"] == "evil.txt"


def test_list_and_get(client):
    a = upload(client, "a.txt", b"aaa content here").json()
    b = upload(client, "b.txt", b"bbb content here").json()
    listed = client.get(BASE).json()
    assert {d["id"] for d in listed} == {a["id"], b["id"]}
    assert client.get(f"{BASE}/{a['id']}").json()["filename"] == "a.txt"


def test_delete_removes_row(client):
    doc = upload(client, "a.txt", b"delete me please").json()
    assert client.delete(f"{BASE}/{doc['id']}").status_code == 204
    assert client.get(f"{BASE}/{doc['id']}").status_code == 404


def test_unknown_and_malformed_ids(client):
    assert client.get(f"{BASE}/00000000-0000-0000-0000-000000000000").status_code == 404
    assert client.get(f"{BASE}/not-a-uuid").status_code == 422


def test_index_endpoint_requires_chunks_first(client):
    # A document that failed chunking (no extractable text) has no chunks to index.
    import io
    from reportlab.pdfgen import canvas as rl_canvas

    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf)
    c.showPage()
    c.save()
    body = upload(client, "blank.pdf", buf.getvalue()).json()
    r = client.post(f"{BASE}/{body['id']}/index")
    assert r.status_code == 409
