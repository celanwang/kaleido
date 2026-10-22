from pathlib import Path

import pytest

from kaleido.deployment.install import verify
from kaleido.deployment.package import package_delivery

ROOT = Path(__file__).resolve().parents[1]


def test_delivery_rejects_an_incomplete_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text(
        '{"kind":"kaleido_cpu_delivery","schema_version":1,"wheel":"missing.whl","files":{}}'
    )
    with pytest.raises(ValueError):
        verify(tmp_path)


def test_packaging_refuses_existing_destinations_and_wrong_distribution(tmp_path):
    import zipfile

    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        package_delivery(
            tmp_path / "x.whl", ROOT / "results/p6/reference/bundle", output, root=ROOT
        )
    wheel = tmp_path / "wrong.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("other.dist-info/METADATA", "Name: other\nVersion: 1.0.0\n")
    with pytest.raises(ValueError, match="name/version"):
        package_delivery(wheel, ROOT / "results/p6/reference/bundle", tmp_path / "new", root=ROOT)


def test_delivery_integrity_detects_a_changed_installer_before_execution(tmp_path):
    import json

    from kaleido.repro.integrity import file_info

    names = [
        "install.py",
        "README.md",
        "pylock.cpu.toml",
        "kaleido.whl",
        "decoder-42/manifest.json",
        "decoder-42/model.pt2",
        "decoder-42/tokenizer.json",
        "decoder-42/examples.json",
        "decoder-42/validation.json",
    ]
    for name in names:
        p = tmp_path / name
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(b"fixed")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "kind": "kaleido_cpu_delivery",
                "schema_version": 1,
                "wheel": "kaleido.whl",
                "files": {n: file_info(tmp_path / n) for n in names},
            }
        )
    )
    verify(tmp_path)
    (tmp_path / "install.py").write_text("changed")
    with pytest.raises(ValueError, match="integrity"):
        verify(tmp_path)
