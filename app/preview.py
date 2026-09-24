"""已完成任务匿名输出实例的 PNG 预览（人工核对影像内容）。

数据来源严格限定为“已完成任务 output_dir 中的匿名 DICOM 文件”，
**不重新解压或处理原始 ZIP**：按调用方给定的**匿名 SOPInstanceUID**
从成功输出索引中唯一定位一份文件后，解码像素并渲染为 8 位灰度 PNG。

支持范围（其余一律明确报错，不静默降级、不猜测选取）：

* 未压缩传输语法（Explicit/Implicit VR Little Endian 等原生像素数据）；
* 单帧（NumberOfFrames 缺省或为 1）；
* 单色图像：PhotometricInterpretation = MONOCHROME1 / MONOCHROME2；
* SamplesPerPixel = 1，BitsAllocated = 8 或 16（有符号/无符号均可）。

渲染管线：

1. 先应用 Rescale Slope / Intercept（缺省分别为 1 / 0）得到 Modality 输出值；
2. 再按 Window Center / Width 线性映射到 8 位灰度（VM>1 时只取首项）；
   缺窗或窗口非法/退化时按像素极值（min/max）线性拉伸；
3. MONOCHROME1 在映射完成后整体反相（255 - y），输出统一为
   MONOCHROME2 观感（0=黑）。
"""
from __future__ import annotations

import collections.abc
import io
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pydicom
from PIL import Image
from pydicom.uid import ImplicitVRLittleEndian

from .anonymizer import FileReport, JobResult
from .study_subset import _build_sop_index, _output_path

# 明确可渲染的未压缩传输语法（PixelData 为原生字节，无需解码器）。
# 未带 TransferSyntaxUID（极少数非标准文件）按 Implicit VR LE 处理。
UNCOMPRESSED_TRANSFER_SYNTAXES = frozenset(
    {
        str(ImplicitVRLittleEndian),
        "1.2.840.10008.1.2.1",  # Explicit VR Little Endian
        "1.2.840.10008.1.2.1.99",  # Deflated Explicit VR Little Endian
        "1.2.840.10008.1.2.2",  # Explicit VR Big Endian (retired)
    }
)

SUPPORTED_PHOTOMETRIC = frozenset({"MONOCHROME1", "MONOCHROME2"})

REASON_AMBIGUOUS_OUTPUT = "ambiguous_output"
REASON_NOT_IN_OUTPUT = "not_in_output"
REASON_UNREADABLE = "unreadable"
REASON_UNSUPPORTED = "unsupported_image"
REASON_DECODE_FAILED = "pixel_decode_failed"
REASON_NO_PIXEL_DATA = "no_pixel_data"


class PreviewError(Exception):
    """预览渲染失败；http_status 由路由层映射为 HTTPException。"""

    def __init__(
        self,
        code: str,
        http_status: int,
        message: str,
        sop_uid: Optional[str] = None,
        reason: Optional[str] = None,
        output_file: Optional[str] = None,
        output_files: Optional[list[str]] = None,
        constraint: Optional[str] = None,
        actual: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        self.detail: dict = {"code": code, "message": message}
        if sop_uid is not None:
            self.detail["sop_instance_uid"] = sop_uid
        if reason is not None:
            self.detail["reason"] = reason
        if output_file is not None:
            self.detail["output_file"] = output_file
        if output_files:
            self.detail["output_files"] = output_files
        if constraint is not None:
            self.detail["constraint"] = constraint
        if actual is not None:
            self.detail["actual"] = actual


@dataclass(frozen=True)
class PreviewResult:
    png: bytes
    rows: int
    columns: int
    bits_allocated: int
    pixel_representation: int
    photometric: str
    rescale_slope: float
    rescale_intercept: float
    window_center: Optional[float]
    window_width: Optional[float]
    window_mode: str  # linear | minmax
    output_file: str


# --------------------------------------------------------------------------- #
# 定位匿名输出
# --------------------------------------------------------------------------- #
def _locate_output(result: JobResult, sop_uid: str) -> FileReport:
    """按匿名 SOP UID 从成功输出中唯一定位；缺失 404、歧义 409、磁盘丢失 409。"""
    targets = _build_sop_index(result).get(sop_uid, [])
    if not targets:
        raise PreviewError(
            "INSTANCE_NOT_FOUND",
            404,
            "该匿名 SOPInstanceUID 在本任务的成功输出中不存在",
            sop_uid=sop_uid,
            reason=REASON_NOT_IN_OUTPUT,
        )
    if len(targets) > 1:
        # 重复实例（单任务内 <uid>.dcm / <uid>-2.dcm）：内容可能不同，
        # 不挑选、不猜测，交由人工决定
        raise PreviewError(
            "AMBIGUOUS_PREVIEW_INSTANCE",
            409,
            "同一匿名 SOPInstanceUID 对应多份输出文件，无法唯一确定预览来源",
            sop_uid=sop_uid,
            reason=REASON_AMBIGUOUS_OUTPUT,
            output_files=sorted(t.output_name for t in targets),
        )
    return targets[0]


# --------------------------------------------------------------------------- #
# 标签校验
# --------------------------------------------------------------------------- #
def _first_float(value: object) -> Optional[float]:
    """取 DICOM 数值元素的首分量（VM>1 只取第一项）；不可解析返回 None。

    pydicom 对多值元素返回 MultiValue（继承 collections.abc.Sequence，
    并非 list 子类）；字符串按标量处理。不依赖 str(value)（多值时为
    "[50, 200]" 形式的 list repr）。
    """
    if value is None:
        return None
    try:
        if (
            isinstance(value, collections.abc.Sequence)
            and not isinstance(value, (str, bytes))
        ):
            if not value:
                return None
            value = value[0]
        text = str(value).strip()
        return float(text) if text else None
    except (TypeError, ValueError):
        return None


def _unsupported(
    message: str,
    *,
    sop_uid: str,
    output_file: str,
    constraint: str,
    actual: object,
) -> PreviewError:
    return PreviewError(
        "UNSUPPORTED_INSTANCE_FOR_PREVIEW",
        422,
        message,
        sop_uid=sop_uid,
        reason=REASON_UNSUPPORTED,
        output_file=output_file,
        constraint=constraint,
        actual="" if actual is None else str(actual),
    )


def _validate_structure(ds, sop_uid: str, output_file: str) -> None:
    """先做纯结构校验（不触发像素解码），压缩/多帧/彩色等一律 422。"""
    meta = getattr(ds, "file_meta", None)
    tsuid = str(getattr(meta, "TransferSyntaxUID", "") or "").strip()
    if tsuid and tsuid not in UNCOMPRESSED_TRANSFER_SYNTAXES:
        raise _unsupported(
            "仅支持未压缩（原生像素数据）传输语法的影像，当前实例使用了压缩/封装传输语法",
            sop_uid=sop_uid,
            output_file=output_file,
            constraint="uncompressed_transfer_syntax",
            actual=tsuid,
        )

    frames = getattr(ds, "NumberOfFrames", None)
    try:
        frame_count = int(str(frames).strip()) if frames not in (None, "") else 1
    except (TypeError, ValueError):
        frame_count = -1
    if frame_count != 1:
        raise _unsupported(
            "仅支持单帧影像（NumberOfFrames 缺省或为 1）",
            sop_uid=sop_uid,
            output_file=output_file,
            constraint="single_frame",
            actual=frames,
        )

    samples = getattr(ds, "SamplesPerPixel", None)
    if samples != 1:
        raise _unsupported(
            "仅支持单色影像（SamplesPerPixel=1），不支持彩色/多采样影像",
            sop_uid=sop_uid,
            output_file=output_file,
            constraint="samples_per_pixel=1",
            actual=samples,
        )

    photometric = str(getattr(ds, "PhotometricInterpretation", "") or "").strip()
    if photometric not in SUPPORTED_PHOTOMETRIC:
        raise _unsupported(
            "仅支持 MONOCHROME1/MONOCHROME2 单色影像",
            sop_uid=sop_uid,
            output_file=output_file,
            constraint="photometric_interpretation in {MONOCHROME1, MONOCHROME2}",
            actual=photometric or None,
        )

    bits_alloc = getattr(ds, "BitsAllocated", None)
    if bits_alloc not in (8, 16):
        raise _unsupported(
            "仅支持 8/16 位影像（BitsAllocated=8 或 16）",
            sop_uid=sop_uid,
            output_file=output_file,
            constraint="bits_allocated in {8, 16}",
            actual=bits_alloc,
        )

    rows = getattr(ds, "Rows", None)
    columns = getattr(ds, "Columns", None)
    if not isinstance(rows, int) or not isinstance(columns, int) or rows <= 0 or columns <= 0:
        raise _unsupported(
            "影像缺少合法的 Rows/Columns，无法确定像素矩阵尺寸",
            sop_uid=sop_uid,
            output_file=output_file,
            constraint="rows>0 and columns>0",
            actual=f"Rows={rows}, Columns={columns}",
        )

    if not getattr(ds, "PixelData", None):
        raise PreviewError(
            "PREVIEW_PIXEL_DATA_MISSING",
            422,
            "实例不含 PixelData，无法渲染影像预览",
            sop_uid=sop_uid,
            reason=REASON_NO_PIXEL_DATA,
            output_file=output_file,
            constraint="pixel_data present",
        )


# --------------------------------------------------------------------------- #
# 像素渲染
# --------------------------------------------------------------------------- #
def _decode_pixels(ds, sop_uid: str, output_file: str) -> np.ndarray:
    try:
        arr = ds.pixel_array
    except Exception as exc:  # 编码异常 / 截断 / 缺少解码器等
        raise PreviewError(
            "PREVIEW_PIXEL_DECODE_FAILED",
            422,
            f"像素数据解码失败：{exc}",
            sop_uid=sop_uid,
            reason=REASON_DECODE_FAILED,
            output_file=output_file,
        ) from exc

    expected = int(ds.Rows) * int(ds.Columns)
    if arr.size != expected:
        raise PreviewError(
            "PREVIEW_PIXEL_DECODE_FAILED",
            422,
            f"解码出的像素数 {arr.size} 与 Rows*Columns={expected} 不一致",
            sop_uid=sop_uid,
            reason=REASON_DECODE_FAILED,
            output_file=output_file,
        )
    return np.asarray(arr, dtype=np.float64).reshape(int(ds.Rows), int(ds.Columns))


def _window_to_gray(
    values: np.ndarray,
    center: Optional[float],
    width: Optional[float],
) -> tuple[np.ndarray, str]:
    """线性窗宽窗位映射到 [0,255]；缺窗/退化窗按极值拉伸。返回 (灰度图, 模式)。"""
    if center is not None and width is not None and width >= 1:
        lower = center - width / 2.0
        gray = (values - lower) * 255.0 / width
        return gray, "linear"

    vmin = float(values.min())
    vmax = float(values.max())
    if vmax <= vmin:
        # 整图常值：极值拉伸未定义（0/0），统一渲染为黑色（MONOCHROME1 随后反相为白）
        return np.zeros(values.shape, dtype=np.float64), "minmax"
    gray = (values - vmin) * 255.0 / (vmax - vmin)
    return gray, "minmax"


def _render(ds, sop_uid: str, output_file: str) -> tuple[np.ndarray, dict]:
    slope = _first_float(getattr(ds, "RescaleSlope", None))
    intercept = _first_float(getattr(ds, "RescaleIntercept", None))
    slope = 1.0 if slope is None else slope
    intercept = 0.0 if intercept is None else intercept

    pixels = _decode_pixels(ds, sop_uid, output_file)
    pixels = pixels * slope + intercept

    center = _first_float(getattr(ds, "WindowCenter", None))
    width = _first_float(getattr(ds, "WindowWidth", None))
    gray, window_mode = _window_to_gray(pixels, center, width)

    gray = np.clip(gray, 0.0, 255.0)
    if str(ds.PhotometricInterpretation).strip() == "MONOCHROME1":
        # MONOCHROME1：最小值映射为最亮；映射到统一 8 位灰度后整体反相
        gray = 255.0 - gray
    image = np.rint(gray).astype(np.uint8)
    return image, {
        "rescale_slope": slope,
        "rescale_intercept": intercept,
        "window_center": center,
        "window_width": width,
        "window_mode": window_mode,
    }


def render_preview(job, sop_uid: str) -> PreviewResult:
    """按匿名 SOP UID 定位成功输出并渲染 8 位灰度 PNG（PNG 字节在内存中生成）。"""
    result: JobResult = job.result
    report = _locate_output(result, sop_uid)
    path = _output_path(result, report)

    try:
        ds = pydicom.dcmread(path)
    except Exception as exc:  # 自己写出的匿名文件不可读：明确报错，不返回占位图
        raise PreviewError(
            "ANONYMIZED_OUTPUT_UNREADABLE",
            422,
            f"无法读取匿名输出文件 {report.output_name}：{exc}",
            sop_uid=sop_uid,
            reason=REASON_UNREADABLE,
            output_file=report.output_name,
        ) from exc

    _validate_structure(ds, sop_uid, report.output_name)
    image, meta = _render(ds, sop_uid, report.output_name)

    buf = io.BytesIO()
    Image.fromarray(image, mode="L").save(buf, format="PNG")
    return PreviewResult(
        png=buf.getvalue(),
        rows=int(ds.Rows),
        columns=int(ds.Columns),
        bits_allocated=int(ds.BitsAllocated),
        pixel_representation=int(getattr(ds, "PixelRepresentation", 0) or 0),
        photometric=str(ds.PhotometricInterpretation).strip(),
        rescale_slope=meta["rescale_slope"],
        rescale_intercept=meta["rescale_intercept"],
        window_center=meta["window_center"],
        window_width=meta["window_width"],
        window_mode=meta["window_mode"],
        output_file=report.output_name,
    )
