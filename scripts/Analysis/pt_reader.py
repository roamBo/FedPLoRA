"""Torch-free reader for PyTorch zip-format state dictionaries."""

from __future__ import annotations

import io
import pickle
import zipfile

import numpy as np


_DTYPE = {
    "FloatStorage": "f4",
    "DoubleStorage": "f8",
    "HalfStorage": "f2",
    "BFloat16Storage": "bf16",
    "LongStorage": "i8",
    "IntStorage": "i4",
}


class _Storage:
    __slots__ = ("key", "dtype", "numel")

    def __init__(self, key: str, dtype: str, numel: int) -> None:
        self.key = key
        self.dtype = dtype
        self.numel = numel


def _load_raw(
    archive: zipfile.ZipFile,
    names: list[str],
    key: str,
    dtype: str,
    numel: int,
) -> np.ndarray:
    candidates = [name for name in names if name.endswith(f"data/{key}")]
    if not candidates:
        raise KeyError(f"storage {key} not found")
    raw = archive.read(candidates[0])
    if dtype == "bf16":
        values = np.frombuffer(raw, dtype="<u2", count=numel)
        return (values.astype(np.uint32) << 16).view(np.float32)
    return np.frombuffer(raw, dtype="<" + dtype, count=numel).copy()


def load_state_dict(path: str):
    """Load a tensor-only state dict as NumPy arrays without importing torch."""

    archive = zipfile.ZipFile(path)
    names = archive.namelist()
    pickle_name = next(name for name in names if name.endswith("data.pkl"))

    class StateUnpickler(pickle.Unpickler):
        def persistent_load(self, pid):
            _, storage_type, key, _, numel = pid
            name = (
                storage_type.__name__
                if hasattr(storage_type, "__name__")
                else str(storage_type)
            )
            return _Storage(str(key), _DTYPE.get(name, "f4"), int(numel))

        def find_class(self, module, name):
            if module.startswith("torch") and name.endswith("Storage"):
                return type(name, (), {"__name__": name})
            if module == "torch._utils" and name in (
                "_rebuild_tensor_v2",
                "_rebuild_tensor",
            ):

                def rebuild(storage, offset, size, stride, *args, **kwargs):
                    flat = _load_raw(
                        archive,
                        names,
                        storage.key,
                        storage.dtype,
                        storage.numel,
                    )
                    count = int(np.prod(size)) if len(size) else 1
                    values = flat[offset : offset + count]
                    return values.reshape(tuple(size)) if len(size) else values

                return rebuild
            if module == "collections" and name == "OrderedDict":
                return dict
            return super().find_class(module, name)

    return StateUnpickler(io.BytesIO(archive.read(pickle_name))).load()
