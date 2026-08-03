from __future__ import annotations

import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

#: Archive extensions we pre-extract when building a media task archive. The
#: sandbox media preamble only unpacks the OUTER *.tar.gz — it does NOT recurse
#: into nested .zip members, so any train.zip/test2.zip left inside the tar
#: would never be opened. We unzip them here so the staged tree yields ready
#: image/audio subdirectories.
_ZIP_SUFFIXES = {".zip"}


def build_task_archive(data_dir: Path, out_path: Path, exclude: set[str] | None = None) -> Path:
    """tar.gz the data_dir tree (arcnames relative to data_dir), skipping any
    file whose name is in `exclude`. Returns out_path."""
    exclude = exclude or set()
    data_dir = Path(data_dir)
    with tarfile.open(out_path, "w:gz") as tf:
        for p in sorted(data_dir.rglob("*")):
            if p.is_file() and p.name not in exclude:
                tf.add(p, arcname=str(p.relative_to(data_dir)))
    return out_path


def build_task_archive_extracted(
    data_dir: Path,
    out_path: Path,
    exclude: set[str] | None = None,
    work_dir: Path | None = None,
) -> Path:
    """Pack a media task tree as tar.gz, PRE-EXTRACTING any nested .zip archives.

    Why: the sandbox media preamble (agent-api/tools/sandbox.py `_MEDIA_HEADER`)
    extracts only the OUTER ``*.tar.gz`` and then lists the resulting top-level
    SUBDIRECTORIES (``_dirs``) as the image/audio dirs and top-level ``*.csv`` as
    label files. If we tar the raw tree verbatim, vision/audio tasks ship their
    media inside nested zips (aerial-cactus: train.zip/test.zip; whale:
    train2.zip/test2.zip), and those zips would land in the workspace unopened —
    the agent would see no media dirs at all.

    This builds a temporary staging dir, then for every member of ``data_dir``:
      - a ``*.zip`` is extracted into a subdirectory of the staging dir named
        after the zip stem (train.zip → train/, train2.zip → train2/). The
        original .zip is NOT copied in (only its contents).
      - any other file (e.g. a label CSV like train.csv, or sampleSubmission)
        is copied to the staging-dir root, UNLESS its name is in ``exclude``.
      - directories already in the tree are copied as-is.
    Finally the staging dir is tarred to ``out_path``.

    The net result, once the preamble extracts the tar.gz, places media files in
    subdirectories (preamble `_dirs`) and any label CSV at the workspace root
    (preamble `_csvs`). Labels are left for the agent to join (aerial-cactus's
    train.csv maps image filename→target; whale has no label CSV — the label is
    encoded in the train2/ filenames and the agent infers it from the prompt).

    Disk note: extraction is done under a temp dir below ``work_dir`` (defaults
    to ``out_path``'s parent) and cleaned up afterwards. The whale zips are
    ~95MB+107MB; peak transient disk is roughly the extracted tree plus the
    final tar.gz, so size ``work_dir`` accordingly.

    Returns ``out_path``.
    """
    exclude = exclude or set()
    data_dir = Path(data_dir)
    out_path = Path(out_path)
    base = Path(work_dir) if work_dir is not None else out_path.parent
    base.mkdir(parents=True, exist_ok=True)

    staging_parent = Path(tempfile.mkdtemp(prefix="bench-stage-", dir=str(base)))
    staging = staging_parent / "task"
    staging.mkdir()
    try:
        for entry in sorted(data_dir.iterdir()):
            if entry.is_dir():
                # Pre-existing media directory — copy verbatim.
                shutil.copytree(entry, staging / entry.name)
                continue
            if not entry.is_file():
                continue
            if entry.name in exclude:
                continue
            if entry.suffix.lower() in _ZIP_SUFFIXES:
                # Unzip into a subdir named after the zip stem so the preamble
                # sees it as a media directory (train.zip → train/).
                target = staging / entry.stem
                target.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(entry) as zf:
                    zf.extractall(target)
                # If the zip already contained a single top-level dir matching
                # its stem, flatten one level so we don't get train/train/...;
                # otherwise leave as-is. (Cheap, best-effort.)
                _maybe_flatten(target)
            else:
                # Label CSV / description / anything else → workspace root.
                shutil.copy2(entry, staging / entry.name)

        # Tar the staging dir's CONTENTS (arcnames relative to staging) so the
        # preamble's top-level _dirs/_csvs land directly in the workspace.
        with tarfile.open(out_path, "w:gz") as tf:
            for p in sorted(staging.rglob("*")):
                if p.is_file():
                    tf.add(p, arcname=str(p.relative_to(staging)))
        return out_path
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)


def _maybe_flatten(target: Path) -> None:
    """If `target` contains exactly one child that is a directory (and nothing
    else), move its contents up one level and remove it. Keeps the staged tree
    from nesting train/train/<files> when a zip already had a top-level folder.
    """
    children = list(target.iterdir())
    if len(children) == 1 and children[0].is_dir():
        inner = children[0]
        for sub in list(inner.iterdir()):
            shutil.move(str(sub), str(target / sub.name))
        inner.rmdir()


def _gcs_client():
    from google.cloud import storage
    return storage.Client()


def stage_archive_direct(sess, sandbox_url: str, session_id: str,
                         archive: Path) -> str:
    """PUT the archive straight into the sandbox session workspace (no GCS).

    Uses the sandbox's raw-bytes upload endpoint
    ``PUT /sessions/{session_id}/files/{filename}`` (sandbox/main.py:445), which
    accepts any size. This is the LOCAL/dev transport: GCS staging needs a
    project + bucket + credentials that a developer box doesn't have, whereas a
    direct PUT works against any reachable sandbox.

    Returns the staged filename.
    """
    archive = Path(archive)
    with archive.open("rb") as fh:
        r = sess.put(
            f"{sandbox_url.rstrip('/')}/sessions/{session_id}/files/{archive.name}",
            data=fh,
            headers={"Content-Type": "application/octet-stream"},
            timeout=(30, 600),
        )
    r.raise_for_status()
    return archive.name


def stage_archive_to_sandbox(sess, sandbox_url: str, session_id: str,
                             archive: Path, bucket: str) -> str:
    """Upload archive to gs://{bucket}/sessions/{sid}/{name}, then POST
    /pull-gcs so the sandbox downloads it into the session workspace.
    Returns the GCS blob path.

    This is the PRODUCTION transport (large archives stream through GCS rather
    than inter-service HTTP). It requires a configured bucket + GCS credentials;
    callers that lack them should use :func:`stage_archive_direct` instead.
    """
    archive = Path(archive)
    blob_path = f"sessions/{session_id}/{archive.name}"
    client = _gcs_client()
    client.bucket(bucket).blob(blob_path).upload_from_filename(str(archive))
    r = sess.post(
        f"{sandbox_url.rstrip('/')}/sessions/{session_id}/pull-gcs",
        json={"bucket": bucket, "blob_path": blob_path, "filename": archive.name},
        timeout=(30, 600),
    )
    r.raise_for_status()
    return blob_path
