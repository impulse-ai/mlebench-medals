"""Prepared MLE-bench dataset inventory for benchmark-first EES."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from bench.adapters.task_detection import AUDIO_EXTS, IMAGE_EXTS, _find_sample_submission


@dataclass(frozen=True)
class DatasetInventory:
    data_dir: Path
    train_csv: Path | None
    test_csv: Path | None
    sample_submission: Path | None
    description_md: Path | None
    csv_files: list[Path] = field(default_factory=list)
    geometry_files: list[Path] = field(default_factory=list)
    image_files: list[Path] = field(default_factory=list)
    audio_files: list[Path] = field(default_factory=list)
    archive_files: list[Path] = field(default_factory=list)
    image_archive_files: list[Path] = field(default_factory=list)
    audio_archive_files: list[Path] = field(default_factory=list)
    total_bytes: int = 0
    mcp_csv_only_safe: bool = False
    mcp_blockers: list[str] = field(default_factory=list)

    @property
    def has_auxiliary_files(self) -> bool:
        return bool(
            self.geometry_files
            or self.image_files
            or self.audio_files
            or self.archive_files
            or self.image_archive_files
            or self.audio_archive_files
        )


def inventory_dataset(
    data_dir: Path,
    max_mcp_upload_bytes: int = 80 * 1024 * 1024,
) -> DatasetInventory:
    """Inventory prepared public data and decide whether CSV-only MCP is safe."""
    data_dir = Path(data_dir)
    train_csv = data_dir / "train.csv" if (data_dir / "train.csv").exists() else None
    test_csv = data_dir / "test.csv" if (data_dir / "test.csv").exists() else None
    sample_submission = _find_sample_submission(data_dir)
    description_md = data_dir / "description.md" if (data_dir / "description.md").exists() else None

    csv_files: list[Path] = []
    geometry_files: list[Path] = []
    image_files: list[Path] = []
    audio_files: list[Path] = []
    archive_files: list[Path] = []
    image_archive_files: list[Path] = []
    audio_archive_files: list[Path] = []
    total_bytes = 0

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        total_bytes += path.stat().st_size
        suffix = path.suffix.lower()
        if suffix == ".csv":
            csv_files.append(path)
        if path.name == "geometry.xyz":
            geometry_files.append(path)
        elif suffix in IMAGE_EXTS:
            image_files.append(path)
        elif suffix in AUDIO_EXTS:
            audio_files.append(path)
        elif suffix in {".zip", ".tar", ".gz", ".tgz", ".bz2", ".7z"}:
            archive_files.append(path)
            if suffix == ".zip":
                archive_kind = _media_kind_in_zip(path)
                if archive_kind == "vision":
                    image_archive_files.append(path)
                elif archive_kind == "audio":
                    audio_archive_files.append(path)

    blockers: list[str] = []
    if geometry_files:
        blockers.append("geometry_files_present")
    if image_files or audio_files or image_archive_files or audio_archive_files:
        blockers.append("media_files_present")
    if archive_files:
        blockers.append("archives_present")
    if train_csv is None:
        blockers.append("missing_train_csv")
    if test_csv is None:
        blockers.append("missing_test_csv")
    for csv_path in [p for p in (train_csv, test_csv) if p is not None]:
        if csv_path.stat().st_size > max_mcp_upload_bytes:
            blockers.append(f"{csv_path.name}_exceeds_mcp_upload_limit")

    return DatasetInventory(
        data_dir=data_dir,
        train_csv=train_csv,
        test_csv=test_csv,
        sample_submission=sample_submission,
        description_md=description_md,
        csv_files=csv_files,
        geometry_files=geometry_files,
        image_files=image_files,
        audio_files=audio_files,
        archive_files=archive_files,
        image_archive_files=image_archive_files,
        audio_archive_files=audio_archive_files,
        total_bytes=total_bytes,
        mcp_csv_only_safe=not blockers,
        mcp_blockers=blockers,
    )


def _media_kind_in_zip(path: Path) -> str | None:
    import zipfile

    has_audio = False
    try:
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                suffix = Path(name).suffix.lower()
                if suffix in IMAGE_EXTS:
                    return "vision"
                if suffix in AUDIO_EXTS:
                    has_audio = True
    except Exception:
        return None
    return "audio" if has_audio else None
