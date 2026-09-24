"""GET /jobs/{job_id}/instances/{sop_instance_uid}/preview.png 端到端测试。"""
from __future__ import annotations

import io
import os
import zipfile

import numpy as np
import pydicom
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    JPEGBaseline8Bit,
    RTStructureSetStorage,
)


# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #
@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DICOM_ANON_WORK_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("DICOM_ANON_JOB_TTL_SECONDS", "3600")
    import importlib
    import app.config
    import app.jobs
    importlib.reload(app.config)
    importlib.reload(app.jobs)
    import app.main
    importlib.reload(app.main)
    with TestClient(app.main.app) as c:
        yield c


def _registry(client):
    from app.main import registry
    return registry


def _image_dcm(
    sop_uid: str,
    *,
    study_uid: str = "1.2.3.4.5.1",
    series_uid: str = "1.2.3.4.5.2",
    patient_id: str = "PV-001",
    photometric: str = "MONOCHROME2",
    bits_allocated: int = 16,
    bits_stored: int | None = None,
    high_bit: int | None = None,
    pixel_representation: int = 0,
    rows: int = 4,
    columns: int = 4,
    pixels: np.ndarray | None = None,
    slope: float | None = None,
    intercept: float | None = None,
    window_center=None,
    window_width=None,
    number_of_frames=None,
    samples_per_pixel: int = 1,
    transfer_syntax=ExplicitVRLittleEndian,
    pixel_data: bytes | None = None,
    with_pixel_data: bool = True,
) -> bytes:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = transfer_syntax
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.PatientID = patient_id
    ds.PatientName = patient_id
    ds.Modality = "CT"
    ds.SamplesPerPixel = samples_per_pixel
    ds.PhotometricInterpretation = photometric
    ds.Rows = rows
    ds.Columns = columns
    ds.BitsAllocated = bits_allocated
    ds.BitsStored = bits_allocated if bits_stored is None else bits_stored
    ds.HighBit = ds.BitsStored - 1 if high_bit is None else high_bit
    ds.PixelRepresentation = pixel_representation
    if number_of_frames is not None:
        ds.NumberOfFrames = number_of_frames
    if slope is not None:
        ds.RescaleSlope = slope
    if intercept is not None:
        ds.RescaleIntercept = intercept
    if window_center is not None:
        ds.WindowCenter = window_center
    if window_width is not None:
        ds.WindowWidth = window_width
    if with_pixel_data:
        if pixel_data is not None:
            ds.PixelData = pixel_data
        else:
            if pixels is None:
                pixels = (np.arange(rows * columns, dtype=np.uint16) * 100).reshape(
                    rows, columns
                )
            if bits_allocated == 8:
                arr = pixels.astype(np.uint8)
            elif pixel_representation == 1:
                arr = pixels.astype(np.int16)
            else:
                arr = pixels.astype(np.uint16)
            ds.PixelData = arr.tobytes()
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


def _zip_bytes(files: dict[str, bytes], path) -> bytes:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, blob in files.items():
            zf.writestr(name, blob)
    return path.read_bytes()


def _upload(client, tmp_path, files, secret="preview-secret-12345"):
    blob = _zip_bytes(files, tmp_path / "in.zip")
    r = client.post(
        "/jobs",
        files={"file": ("study.zip", blob, "application/zip")},
        data={"secret": secret, "shift_dates": "false"},
    )
    assert r.status_code == 201, r.text
    return r.json()


def _anon_sop(client, job_id: str, original_sop: str) -> str:
    job = _registry(client).get(job_id)
    return job.result.sops[original_sop]


def _png_array(response) -> np.ndarray:
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/png"
    img = Image.open(io.BytesIO(response.content))
    assert img.format == "PNG"
    return np.asarray(img)


# --------------------------------------------------------------------------- #
# 成功路径
# --------------------------------------------------------------------------- #
def test_preview_output_file_missing_treated_as_absent(client, tmp_path):
    # 成功输出索引只包含磁盘上仍存在的文件：文件被删除后按“实例不在成功输出”处理
    sop = "1.2.3.4.5.23"
    files = {"img.dcm": _image_dcm(sop, rows=2, columns=2)}
    body = _upload(client, tmp_path, files)
    job = _registry(client).get(body["id"])
    report = next(r for r in job.result.files if r.sop_instance_uid)
    os.unlink(os.path.join(job.result.output_dir, report.output_name))

    r = client.get(
        f"/jobs/{body['id']}/instances/{_anon_sop(client, body['id'], sop)}/preview.png"
    )
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert detail["code"] == "INSTANCE_NOT_FOUND"
    assert detail["reason"] == "not_in_output"


def test_preview_16bit_monochrome2_with_window(client, tmp_path):
    # slope=1, intercept=0；窗中心 500、窗宽 1000 -> 窗界 [0,1000]
    sop = "1.2.3.4.5.9"
    pixels = np.array(
        [[0, 100, 200, 400],
         [600, 800, 900, 1000],
         [100, 300, 500, 700],
         [0, 250, 750, 1000]],
        dtype=np.uint16,
    )
    files = {
        "img.dcm": _image_dcm(
            sop, pixels=pixels, slope=1.0, intercept=0.0,
            window_center=500, window_width=1000,
        ),
    }
    body = _upload(client, tmp_path, files)
    new_sop = _anon_sop(client, body["id"], sop)

    r = client.get(f"/jobs/{body['id']}/instances/{new_sop}/preview.png")
    img = _png_array(r)
    assert img.shape == (4, 4)
    assert img.dtype == np.uint8
    assert r.headers["x-preview-window-mode"] == "linear"
    assert r.headers["x-preview-rescale-slope"] == "1.0"
    assert r.headers["x-preview-rescale-intercept"] == "0.0"
    assert r.headers["x-preview-photometric-interpretation"] == "MONOCHROME2"
    assert r.headers["x-preview-bits-allocated"] == "16"
    assert "inline" in r.headers["content-disposition"]
    assert new_sop[-24:] in r.headers["content-disposition"]

    expected = np.rint(pixels.astype(np.float64) * 255.0 / 1000.0).astype(np.uint8)
    np.testing.assert_array_equal(img, expected)
    assert img[0, 0] == 0       # 窗下界
    assert img[1, 3] == 255     # 窗上界
    assert img[2, 2] == 128     # 窗中心附近 500 -> 127.5 -> 128


def test_preview_window_endpoint_mapping_check(client, tmp_path):
    # 独立精确核对：窗中心 500、窗宽 400，slope=2 intercept=100
    sop = "1.2.3.4.5.10"
    pixels = np.array([[0, 100, 200, 300]], dtype=np.uint16)
    files = {"img.dcm": _image_dcm(
        sop, rows=1, columns=4, pixels=pixels,
        slope=2, intercept=100, window_center=500, window_width=400,
    )}
    body = _upload(client, tmp_path, files)
    r = client.get(
        f"/jobs/{body['id']}/instances/{_anon_sop(client, body['id'], sop)}/preview.png"
    )
    img = _png_array(r)
    # 输出值：100, 300, 500, 700；窗 [300,700]
    expected_vals = [0.0, 0.0, 255.0 / 2.0, 255.0]
    np.testing.assert_array_equal(
        img[0], np.rint(np.array(expected_vals)).astype(np.uint8)
    )


def test_preview_8bit_unsigned(client, tmp_path):
    sop = "1.2.3.4.5.11"
    pixels = np.array([[0, 64, 128, 255]], dtype=np.uint8)
    files = {"img.dcm": _image_dcm(
        sop, rows=1, columns=4, bits_allocated=8, pixels=pixels,
    )}
    body = _upload(client, tmp_path, files)
    r = client.get(
        f"/jobs/{body['id']}/instances/{_anon_sop(client, body['id'], sop)}/preview.png"
    )
    img = _png_array(r)
    # 无窗：minmax 拉伸 0..255 -> 0..255，保持原值
    np.testing.assert_array_equal(img[0], np.array([0, 64, 128, 255], dtype=np.uint8))
    assert r.headers["x-preview-window-mode"] == "minmax"


def test_preview_16bit_signed_rescale(client, tmp_path):
    sop = "1.2.3.4.5.12"
    pixels = np.array([[-1000, 0, 1000]], dtype=np.int16)
    files = {"img.dcm": _image_dcm(
        sop, rows=1, columns=3, pixel_representation=1, pixels=pixels,
        slope=1, intercept=0,
    )}
    body = _upload(client, tmp_path, files)
    r = client.get(
        f"/jobs/{body['id']}/instances/{_anon_sop(client, body['id'], sop)}/preview.png"
    )
    img = _png_array(r)
    # minmax 拉伸 -1000..1000
    expected = np.rint(
        (pixels.astype(np.float64) - (-1000.0)) * 255.0 / 2000.0
    ).astype(np.uint8)
    np.testing.assert_array_equal(img.reshape(-1), expected.reshape(-1))
    assert r.headers["x-preview-pixel-representation"] == "1"


def test_preview_monochrome1_is_inverted(client, tmp_path):
    sop = "1.2.3.4.5.13"
    pixels = np.array([[0, 85, 170, 255]], dtype=np.uint8)
    files = {"m1.dcm": _image_dcm(
        sop, rows=1, columns=4, bits_allocated=8,
        photometric="MONOCHROME1", pixels=pixels,
    )}
    body = _upload(client, tmp_path, files)
    r = client.get(
        f"/jobs/{body['id']}/instances/{_anon_sop(client, body['id'], sop)}/preview.png"
    )
    img = _png_array(r)
    # MONOCHROME1 极值拉伸后整体反相
    expected = 255 - np.array([0, 85, 170, 255], dtype=np.uint8)
    np.testing.assert_array_equal(img[0], expected)
    assert r.headers["x-preview-photometric-interpretation"] == "MONOCHROME1"


def test_preview_multivalue_window_uses_first_item(client, tmp_path):
    sop = "1.2.3.4.5.14"
    pixels = np.array([[0, 100, 200, 400]], dtype=np.uint16)
    # 通过 DS 多值元素构造 VM=2 的窗（pydicom 会正确写为反斜杠分隔）
    from pydicom.dataelem import DataElement

    blob = _image_dcm(sop, rows=1, columns=4, pixels=pixels)
    ds = pydicom.dcmread(io.BytesIO(blob))
    ds[0x00281050] = DataElement(0x00281050, "DS", ["50", "200"])
    ds[0x00281051] = DataElement(0x00281051, "DS", ["100", "400"])
    out = io.BytesIO()
    ds.save_as(out, enforce_file_format=True)
    body = _upload(client, tmp_path, {"img.dcm": out.getvalue()})
    r = client.get(
        f"/jobs/{body['id']}/instances/{_anon_sop(client, body['id'], sop)}/preview.png"
    )
    img = _png_array(r)
    assert r.headers["x-preview-window-mode"] == "linear"
    # 首项 center=50 width=100 -> 窗 [0,100]
    values = pixels.astype(np.float64)
    expected = np.rint(np.clip((values - 0.0) * 255.0 / 100.0, 0, 255)).astype(np.uint8)
    np.testing.assert_array_equal(img.reshape(-1), expected.reshape(-1))


def test_preview_missing_window_stretches_to_extrema(client, tmp_path):
    sop = "1.2.3.4.5.15"
    pixels = np.array([[1000, 1100, 1200, 2000]], dtype=np.uint16)
    files = {"img.dcm": _image_dcm(sop, rows=1, columns=4, pixels=pixels)}
    body = _upload(client, tmp_path, files)
    r = client.get(
        f"/jobs/{body['id']}/instances/{_anon_sop(client, body['id'], sop)}/preview.png"
    )
    img = _png_array(r)
    assert r.headers["x-preview-window-mode"] == "minmax"
    expected = np.rint(
        (pixels.astype(np.float64) - 1000.0) * 255.0 / 1000.0
    ).astype(np.uint8)
    np.testing.assert_array_equal(img.reshape(-1), expected.reshape(-1))
    np.testing.assert_array_equal(img.reshape(-1)[[0, -1]], np.array([0, 255], dtype=np.uint8))


def test_preview_zero_width_window_falls_back_to_minmax(client, tmp_path):
    sop = "1.2.3.4.5.16"
    pixels = np.array([[10, 20]], dtype=np.uint16)
    files = {"img.dcm": _image_dcm(
        sop, rows=1, columns=2, pixels=pixels, window_center=15, window_width=0,
    )}
    body = _upload(client, tmp_path, files)
    r = client.get(
        f"/jobs/{body['id']}/instances/{_anon_sop(client, body['id'], sop)}/preview.png"
    )
    img = _png_array(r)
    assert r.headers["x-preview-window-mode"] == "minmax"


def test_preview_partial_window_falls_back_to_minmax(client, tmp_path):
    # 只给 WindowCenter 不给 WindowWidth：视为缺窗，按极值拉伸
    sop = "1.2.3.4.5.19"
    pixels = np.array([[10, 20, 30]], dtype=np.uint16)
    files = {"img.dcm": _image_dcm(
        sop, rows=1, columns=3, pixels=pixels, window_center=20,
    )}
    body = _upload(client, tmp_path, files)
    r = client.get(
        f"/jobs/{body['id']}/instances/{_anon_sop(client, body['id'], sop)}/preview.png"
    )
    img = _png_array(r)
    assert r.headers["x-preview-window-mode"] == "minmax"
    assert "x-preview-window-width" not in r.headers


def test_preview_12bit_stored_in_16bit_container(client, tmp_path):
    # 12 位像素存储在 16 位容器（BitsAllocated=16, BitsStored=12），缺窗极值拉伸
    sop = "1.2.3.4.5.22"
    pixels = np.array([[0, 2048, 4095]], dtype=np.uint16)
    files = {"img.dcm": _image_dcm(
        sop, rows=1, columns=3, bits_allocated=16, bits_stored=12, high_bit=11,
        pixels=pixels,
    )}
    body = _upload(client, tmp_path, files)
    r = client.get(
        f"/jobs/{body['id']}/instances/{_anon_sop(client, body['id'], sop)}/preview.png"
    )
    img = _png_array(r)
    np.testing.assert_array_equal(
        img.reshape(-1), np.array([0, 128, 255], dtype=np.uint8)
    )


def test_preview_constant_image_renders(client, tmp_path):
    sop = "1.2.3.4.5.17"
    pixels = np.full((2, 2), 42, dtype=np.uint16)
    files = {"img.dcm": _image_dcm(sop, rows=2, columns=2, pixels=pixels)}
    body = _upload(client, tmp_path, files)
    r = client.get(
        f"/jobs/{body['id']}/instances/{_anon_sop(client, body['id'], sop)}/preview.png"
    )
    img = _png_array(r)
    assert img.shape == (2, 2)
    assert set(np.unique(img)) == {0}


def test_preview_rows_columns_headers(client, tmp_path):
    sop = "1.2.3.4.5.18"
    pixels = (np.arange(3 * 5, dtype=np.uint16) * 50).reshape(3, 5)
    files = {"img.dcm": _image_dcm(sop, rows=3, columns=5, pixels=pixels)}
    body = _upload(client, tmp_path, files)
    r = client.get(
        f"/jobs/{body['id']}/instances/{_anon_sop(client, body['id'], sop)}/preview.png"
    )
    assert r.status_code == 200
    assert r.headers["x-preview-rows"] == "3"
    assert r.headers["x-preview-columns"] == "5"


def test_preview_implicit_vr_little_endian(client, tmp_path):
    # Implicit VR Little Endian 同样属于未压缩语法，应正常渲染
    from pydicom.uid import ImplicitVRLittleEndian

    sop = "1.2.3.4.5.24"
    pixels = np.array([[0, 100, 200, 300]], dtype=np.uint16)
    files = {"img.dcm": _image_dcm(
        sop, rows=1, columns=4, pixels=pixels,
        transfer_syntax=ImplicitVRLittleEndian,
    )}
    body = _upload(client, tmp_path, files)
    r = client.get(
        f"/jobs/{body['id']}/instances/{_anon_sop(client, body['id'], sop)}/preview.png"
    )
    img = _png_array(r)
    expected = np.rint(pixels.astype(np.float64) * 255.0 / 300.0).astype(np.uint8)
    np.testing.assert_array_equal(img.reshape(-1), expected.reshape(-1))


# --------------------------------------------------------------------------- #
# 定位 / 冲突
# --------------------------------------------------------------------------- #
def test_preview_unknown_sop_404(client, tmp_path):
    files = {"img.dcm": _image_dcm("1.2.3.4.5.20", rows=2, columns=2)}
    body = _upload(client, tmp_path, files)
    r = client.get(f"/jobs/{body['id']}/instances/2.25.9999999999/preview.png")
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert detail["code"] == "INSTANCE_NOT_FOUND"
    assert detail["reason"] == "not_in_output"
    assert detail["sop_instance_uid"] == "2.25.9999999999"


def test_preview_job_missing_404(client):
    r = client.get("/jobs/nope/instances/2.25.1/preview.png")
    assert r.status_code == 404


def test_preview_duplicate_sop_outputs_conflict_409(client, tmp_path):
    # 两个输入文件共享同一 SOPInstanceUID、像素内容不同 -> 输出改名两份
    sop = "1.2.3.4.5.21"
    p1 = np.array([[0, 0], [0, 0]], dtype=np.uint16)
    p2 = np.array([[100, 200], [300, 400]], dtype=np.uint16)
    files = {
        "a/dup.dcm": _image_dcm(sop, pixels=p1),
        "b/dup.dcm": _image_dcm(sop, pixels=p2),
    }
    body = _upload(client, tmp_path, files)
    job = _registry(client).get(body["id"])
    new_sop = job.result.sops[sop]
    outputs = [f["output_name"] for f in body["files"] if f["sop_instance_uid"] == new_sop]
    assert len(outputs) == 2  # <new_sop>.dcm 与 <new_sop>-2.dcm

    r = client.get(f"/jobs/{body['id']}/instances/{new_sop}/preview.png")
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["code"] == "AMBIGUOUS_PREVIEW_INSTANCE"
    assert detail["reason"] == "ambiguous_output"
    assert sorted(detail["output_files"]) == sorted(outputs)
    assert detail["sop_instance_uid"] == new_sop


def test_preview_failed_job_409(client):
    body = client.post(
        "/jobs",
        files={"file": ("broken.zip", b"PK\x03\x04 garbage", "application/zip")},
        data={"secret": "preview-secret-12345"},
    ).json()
    r = client.get(f"/jobs/{body['id']}/instances/2.25.1/preview.png")
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "JOB_NOT_READY"


def test_preview_non_image_sop_is_422(client, tmp_path):
    # RTSTRUCT 没有像素：不支持预览（不能静默返回别的文件）
    img_sop = "1.2.3.4.5.30"
    rt_sop = "1.2.3.4.5.31"
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = RTStructureSetStorage
    fm.MediaStorageSOPInstanceUID = rt_sop
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=fm, preamble=b"\0" * 128)
    ds.SOPClassUID = RTStructureSetStorage
    ds.SOPInstanceUID = rt_sop
    ds.StudyInstanceUID = "1.2.3.4.5.1"
    ds.SeriesInstanceUID = "1.2.3.4.5.7"
    ds.PatientID = "PV-001"
    ds.PatientName = "PV-001"
    ds.Modality = "RTSTRUCT"
    ds.StructureSetLabel = "RT"
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)

    files = {"img.dcm": _image_dcm(img_sop, rows=2, columns=2), "rt.dcm": buf.getvalue()}
    body = _upload(client, tmp_path, files)
    new_rt = _anon_sop(client, body["id"], rt_sop)
    r = client.get(f"/jobs/{body['id']}/instances/{new_rt}/preview.png")
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["code"] == "UNSUPPORTED_INSTANCE_FOR_PREVIEW"
    assert detail["reason"] == "unsupported_image"


# --------------------------------------------------------------------------- #
# 不支持的影像
# --------------------------------------------------------------------------- #
def _expect_unsupported(client, body, sop, constraint):
    new_sop = _anon_sop(client, body["id"], sop)
    r = client.get(f"/jobs/{body['id']}/instances/{new_sop}/preview.png")
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert detail["code"] == "UNSUPPORTED_INSTANCE_FOR_PREVIEW"
    assert detail["constraint"] == constraint
    return detail


def test_preview_compressed_transfer_syntax_422(client, tmp_path):
    sop = "1.2.3.4.5.40"
    # 先按未压缩语法写出，再把传输语法改为 JPEG Baseline 并写入封装像素数据
    # （构造一个声明压缩语法的实例；结构校验应先于像素解码拒绝）
    from pydicom.encaps import encapsulate

    blob = _image_dcm(sop, rows=2, columns=2, bits_allocated=8)
    ds = pydicom.dcmread(io.BytesIO(blob))
    ds.file_meta.TransferSyntaxUID = JPEGBaseline8Bit
    ds.PixelData = encapsulate(
        [b"\xff\xd8\xff\xe0not-a-real-jpeg-frame\xff\xd9"], has_bot=False
    )
    out = io.BytesIO()
    ds.save_as(out, enforce_file_format=True)
    body = _upload(client, tmp_path, {"img.dcm": out.getvalue()})
    detail = _expect_unsupported(client, body, sop, "uncompressed_transfer_syntax")
    assert detail["actual"] == str(JPEGBaseline8Bit)


def test_preview_multiframe_422(client, tmp_path):
    sop = "1.2.3.4.5.41"
    pixels = np.zeros((2, 2, 2), dtype=np.uint16)
    blob = _image_dcm(
        sop, number_of_frames=2, pixel_data=pixels.tobytes()
    )
    body = _upload(client, tmp_path, {"img.dcm": blob})
    _expect_unsupported(client, body, sop, "single_frame")


def test_preview_color_422(client, tmp_path):
    sop = "1.2.3.4.5.42"
    pixels = np.zeros((2, 2, 3), dtype=np.uint8)
    blob = _image_dcm(
        sop, bits_allocated=8, samples_per_pixel=3,
        photometric="RGB", pixel_data=pixels.tobytes(),
    )
    body = _upload(client, tmp_path, {"img.dcm": blob})
    _expect_unsupported(client, body, sop, "samples_per_pixel=1")


def test_preview_32bit_422(client, tmp_path):
    sop = "1.2.3.4.5.43"
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = CTImageStorage
    fm.MediaStorageSOPInstanceUID = sop
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=fm, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = sop
    ds.StudyInstanceUID = "1.2.3.4.5.1"
    ds.SeriesInstanceUID = "1.2.3.4.5.2"
    ds.PatientID = "PV-001"
    ds.PatientName = "PV-001"
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = 2
    ds.Columns = 2
    ds.BitsAllocated = 32
    ds.BitsStored = 32
    ds.HighBit = 31
    ds.PixelRepresentation = 0
    ds.PixelData = np.zeros(4, dtype=np.uint32).tobytes()
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    body = _upload(client, tmp_path, {"img.dcm": buf.getvalue()})
    _expect_unsupported(client, body, sop, "bits_allocated in {8, 16}")


def test_preview_palette_color_photometric_422(client, tmp_path):
    sop = "1.2.3.4.5.44"
    blob = _image_dcm(
        sop, bits_allocated=8, photometric="PALETTE COLOR",
        samples_per_pixel=1, pixels=np.zeros((2, 2), dtype=np.uint8),
    )
    body = _upload(client, tmp_path, {"img.dcm": blob})
    _expect_unsupported(
        client, body, sop,
        "photometric_interpretation in {MONOCHROME1, MONOCHROME2}",
    )


def test_preview_without_pixel_data_422(client, tmp_path):
    sop = "1.2.3.4.5.45"
    blob = _image_dcm(sop, with_pixel_data=False)
    body = _upload(client, tmp_path, {"img.dcm": blob})
    new_sop = _anon_sop(client, body["id"], sop)
    r = client.get(f"/jobs/{body['id']}/instances/{new_sop}/preview.png")
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "PREVIEW_PIXEL_DATA_MISSING"


def test_preview_truncated_pixel_data_422(client, tmp_path):
    sop = "1.2.3.4.5.46"
    # Rows*Columns=16 个 16 位像素需要 32 字节，只给 8 字节
    blob = _image_dcm(sop, pixel_data=b"\x00" * 8)
    body = _upload(client, tmp_path, {"img.dcm": blob})
    new_sop = _anon_sop(client, body["id"], sop)
    r = client.get(f"/jobs/{body['id']}/instances/{new_sop}/preview.png")
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["code"] == "PREVIEW_PIXEL_DECODE_FAILED"
    assert detail["reason"] == "pixel_decode_failed"
