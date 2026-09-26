import os.path as op
import shutil
import struct
from io import BytesIO
from pathlib import Path
from typing import Literal

import pytest
from utils import get_files, plat_map

from hgpaktool import HGPAKFile
from hgpaktool.api import FILEINFO_FMT, HGPakFileIndex, InvalidFileException
from hgpaktool.utils import normalise_path, parse_manifest

DATA_DIR = op.join(op.dirname(__file__), "data")


@pytest.mark.parametrize("platform", ("windows", "mac", "linux"))
def test_unpack(tmp_path: Path, platform: Literal["windows", "mac", "linux"]):
    with HGPAKFile(op.join(DATA_DIR, f"NMSARC.MeshPlanetSKY.{platform}.pak"), platform) as pak:
        print(pak.filenames)
        assert len(pak.filenames) == 6
        # Extract the files to a temporary directory and analyse it
        pak.unpack(tmp_path)
        assert len(get_files(tmp_path)) == 6
        shutil.rmtree(tmp_path / "models")
        pak.unpack(tmp_path, upper=True)
        files = get_files(tmp_path)
        assert len(files) == 6
        # For each file ensure the base path component up to the extraction path has its capitalization
        # retained. Then check the filename itself is uppercase.
        for fpath in files:
            assert fpath.startswith(str(tmp_path))
            final_path = Path(fpath).relative_to(tmp_path)
            assert str(final_path).upper() == str(final_path)


@pytest.mark.parametrize("platform", ("windows", "mac", "linux"))
def test_unpack_with_manifest(tmp_path: Path, platform: Literal["windows", "mac", "linux"]):
    with HGPAKFile(op.join(DATA_DIR, f"NMSARC.MeshPlanetSKY.{platform}.pak"), platform) as pak:
        assert len(pak.filenames) == 6
        # Extract the files to a temporary directory and analyse it
        pak.unpack(tmp_path, write_manifest=True)
        files = get_files(tmp_path)
        assert len(files) == 7
        # Find the manifest file
        manifest_fpath = None
        for fpath in files:
            if fpath.endswith(".manifest"):
                manifest_fpath = fpath
        assert manifest_fpath is not None
        assert op.exists(manifest_fpath)

        manifest_contents = parse_manifest(manifest_fpath)
        # Remove the base path and normalise the paths of the real files and check that they match the
        # manifest
        norm_paths = []
        for fpath in files:
            if fpath != manifest_fpath:
                norm_paths.append(normalise_path(op.relpath(fpath, tmp_path)))
        assert len(norm_paths) == 6
        assert set(norm_paths) == set(manifest_contents)


@pytest.mark.parametrize("platform", ("windows", "mac", "linux"))
def test_filtered_extraction(platform: Literal["windows", "mac", "linux"]):
    with HGPAKFile(op.join(DATA_DIR, f"NMSARC.MeshPlanetSKY.{platform}.pak"), platform) as pak:
        assert len([x for x in pak.extract("*rainbowplane*")]) == 2
        assert len([x for x in pak.extract("*RAINBOWPLANE*")]) == 2
        assert len([x for x in pak.extract(["*rainbowplane*", "*skycube*"])]) == 4
        assert (
            len([x for x in pak.extract(f"models/planets/sky/skysphere.geometry.mbin.{plat_map[platform]}")])
            == 1
        )
        assert (
            len(
                [
                    x
                    for x in pak.extract(
                        f"MODELS/PLANETS/SKY/SKYSPHERE.GEOMETRY.MBIN.{plat_map[platform].upper()}"
                    )
                ]
            )
            == 1
        )


@pytest.mark.parametrize("platform", ("windows", "mac", "linux"))
@pytest.mark.parametrize("as_buffer", (True, False))
def test_extract_specific_single_file(platform: Literal["windows", "mac", "linux"], as_buffer: bool):
    pak = HGPAKFile(op.join(DATA_DIR, f"NMSARC.MeshPlanetSKY.{platform}.pak"), platform)
    fname = f"models/planets/sky/skysphere.geometry.mbin.{plat_map[platform]}"
    data = pak.extract_specific(fname, as_buffer)
    if as_buffer:
        assert isinstance(data, BytesIO)
        assert len(data.getvalue()) > 0
    else:
        assert isinstance(data, bytes)
        assert len(data) > 0


@pytest.mark.parametrize("platform", ("windows", "mac", "linux"))
@pytest.mark.parametrize("as_buffer", (True, False))
def test_extract_specific_multi_file(platform: Literal["windows", "mac", "linux"], as_buffer: bool):
    pak = HGPAKFile(op.join(DATA_DIR, f"NMSARC.MeshPlanetSKY.{platform}.pak"), platform)
    fnames = [
        f"models/planets/sky/skysphere.geometry.mbin.{plat_map[platform]}",
        f"models/planets/sky/rainbowplane.geometry.mbin.{plat_map[platform]}",
    ]
    data = pak.extract_specific(fnames, as_buffer)
    assert isinstance(data, dict)
    assert set(data.keys()) == set(fnames)
    if as_buffer:
        for blob in data.values():
            assert isinstance(blob, BytesIO)
            assert len(blob.getvalue()) > 0
    else:
        for blob in data.values():
            assert isinstance(blob, bytes)
            assert len(blob) > 0


@pytest.mark.parametrize("platform", ("windows", "mac", "linux"))
def test_filtered_extraction_mixed_filter_shapes_combine_correctly(
    platform: Literal["windows", "mac", "linux"],
):
    """_get_filtered_filelist() was rewritten for speed against real data with many filters in one call
    (see that method's own doc comment) into three separate paths - plain substring containment for the
    common "*text*" shape, a combined compiled regex for anything with other glob metacharacters, and a
    direct dict entry for an exact (non-wildcard) filter - this confirms all three combine correctly in
    one call, matching what running each filter's old separate fnmatch.filter()/exact-lookup would have
    produced."""
    with HGPAKFile(op.join(DATA_DIR, f"NMSARC.MeshPlanetSKY.{platform}.pak"), platform) as pak:
        exact_name = f"models/planets/sky/skysphere.geometry.mbin.{plat_map[platform]}"
        filters = [
            "*rainbowplane*",  # plain substring shape
            "models/planets/sky/skycube*",  # general glob (no leading "*")
            exact_name,  # exact, non-wildcard
        ]
        matched = set(pak._get_filtered_filelist(filters))
        assert exact_name in matched
        assert any("rainbowplane" in name for name in matched)
        assert any(name.startswith("models/planets/sky/skycube") for name in matched)
        # Exactly the union of what each filter alone would match - nothing extra, nothing missing.
        expected = (
            set(pak._get_filtered_filelist("*rainbowplane*"))
            | set(pak._get_filtered_filelist("models/planets/sky/skycube*"))
            | {exact_name}
        )
        assert matched == expected


def test_file_index_read_bulk_matches_per_entry_semantics():
    """HGPakFileIndex.read() was rewritten to bulk-read+decode the whole file index in one shot instead
    of one `fobj.read()` + `struct.unpack()` per entry (a perf fix for paks with many thousands of
    files) - this confirms the bulk decode still produces identical FileInfo entries, and that the `n`
    partial-read parameter still stops after exactly n entries and leaves the stream positioned right
    after them (rather than after the full file_count), since callers rely on that to keep reading
    whatever comes next in the file."""
    entries = [
        (b"\x01" * 16, 0x100, 0x10),
        (b"\x02" * 16, 0x200, 0x20),
        (b"\x03" * 16, 0x300, 0x30),
    ]
    raw = b"".join(struct.pack(FILEINFO_FMT, *entry) for entry in entries)

    # Full read (n=-1, the default): every entry decoded, in order.
    index = HGPakFileIndex()
    index.read(len(entries), BytesIO(raw))
    assert [finf.values() for finf in index.fileInfo] == entries

    # Partial read (n < file_count): stops after exactly n entries, leaving the stream positioned right
    # after them, not after the full file_count.
    fobj = BytesIO(raw + b"TRAILING")
    index = HGPakFileIndex()
    index.read(len(entries), fobj, n=2)
    assert [finf.values() for finf in index.fileInfo] == entries[:2]
    assert fobj.read() == struct.pack(FILEINFO_FMT, *entries[2]) + b"TRAILING"


def test_invalid_pak():
    with pytest.raises(InvalidFileException):
        with HGPAKFile(op.join(DATA_DIR, "NMSARC.MeshPlanetSKY.invalid.pak")):
            pass


# TODO: Add test for switch pak's.
