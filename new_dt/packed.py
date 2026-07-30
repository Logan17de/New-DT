from __future__ import annotations

import json
import math
import mmap
import os
import struct
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .routing import SelectivePageReconstructionStore


_MAGIC = b"SPRCB001"
_HEADER = struct.Struct("<8sQ")
_VERSION = 1


def bits_required(max_value: int) -> int:
    return max(1, int(max_value).bit_length())


def pack_uints(values: Iterable[int], bit_width: int) -> bytes:
    """Pack unsigned integers exactly using a fixed arbitrary bit width."""

    if not 1 <= bit_width <= 64:
        raise ValueError("bit_width must be in [1, 64]")
    mask = (1 << bit_width) - 1
    output = bytearray()
    accumulator = 0
    available = 0
    for raw in values:
        value = int(raw)
        if value < 0 or value > mask:
            raise ValueError(f"value {value} does not fit in {bit_width} bits")
        accumulator |= value << available
        available += bit_width
        while available >= 8:
            output.append(accumulator & 0xFF)
            accumulator >>= 8
            available -= 8
    if available:
        output.append(accumulator & 0xFF)
    return bytes(output)


def unpack_uints(data: bytes | bytearray | memoryview, count: int, bit_width: int) -> list[int]:
    if count < 0:
        raise ValueError("count must be non-negative")
    return [packed_uint_at(data, index, bit_width) for index in range(count)]


def packed_uint_at(
    data: bytes | bytearray | memoryview | mmap.mmap,
    index: int,
    bit_width: int,
    *,
    byte_offset: int = 0,
) -> int:
    """Read one exact value without decoding neighboring packed values."""

    if index < 0:
        raise IndexError("packed integer index must be non-negative")
    bit_position = index * bit_width
    first_byte = byte_offset + bit_position // 8
    shift = bit_position % 8
    byte_count = (shift + bit_width + 7) // 8
    raw = int.from_bytes(data[first_byte : first_byte + byte_count], "little")
    return (raw >> shift) & ((1 << bit_width) - 1)


def encode_uvarint(value: int) -> bytes:
    if value < 0:
        raise ValueError("uvarint cannot encode a negative value")
    output = bytearray()
    while value >= 0x80:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)
    return bytes(output)


def decode_uvarint(data: bytes | bytearray | memoryview | mmap.mmap, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
        if shift > 63:
            raise ValueError("invalid or oversized uvarint")


def _encode_patch(entries: Iterable[tuple[int, int]]) -> bytes:
    sorted_entries = sorted((int(offset), int(scalar)) for offset, scalar in entries)
    output = bytearray(encode_uvarint(len(sorted_entries)))
    previous = 0
    for index, (offset, scalar) in enumerate(sorted_entries):
        delta = offset if index == 0 else offset - previous
        if delta < 0:
            raise ValueError("patch offsets must be non-decreasing")
        output.extend(encode_uvarint(delta))
        output.extend(encode_uvarint(scalar))
        previous = offset
    return bytes(output)


def _decode_patch(
    data: bytes | bytearray | memoryview | mmap.mmap,
    start: int,
    end: int,
) -> tuple[dict[int, int], int]:
    count, cursor = decode_uvarint(data, start)
    entries: dict[int, int] = {}
    previous = 0
    for index in range(count):
        delta, cursor = decode_uvarint(data, cursor)
        scalar, cursor = decode_uvarint(data, cursor)
        offset = delta if index == 0 else previous + delta
        entries[offset] = scalar
        previous = offset
    if cursor > end:
        raise ValueError("packed patch exceeds its indexed range")
    return entries, cursor


class PackedSPRCWriter:
    """Serialize an SPRC route program into one exact independently readable file."""

    @staticmethod
    def write(
        store: SelectivePageReconstructionStore,
        path: str | os.PathLike[str],
        *,
        pool_size: int,
    ) -> dict[str, int]:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = bytearray()

        neuron_bits = bits_required(max(0, pool_size - 1))
        selector_state = store.selector_state()
        defaults = selector_state["token_defaults"]
        if not torch.is_tensor(defaults):
            raise TypeError("selector defaults must be a tensor")
        selector_bits = bits_required(store.max_template_id())
        default_bytes = pack_uints(defaults.tolist(), selector_bits)
        selector_default_offset = len(payload)
        payload.extend(default_bytes)

        selector_override_entries: list[list[int]] = []
        for page_id in store.selector_override_pages():
            overrides = store.selector_page_overrides(page_id)
            encoded = bytearray(encode_uvarint(len(overrides)))
            previous_token = 0
            for index, (token_id, template_id) in enumerate(sorted(overrides.items())):
                token_delta = token_id if index == 0 else token_id - previous_token
                encoded.extend(encode_uvarint(token_delta))
                encoded.extend(encode_uvarint(template_id))
                previous_token = token_id
            offset = len(payload)
            payload.extend(encoded)
            selector_override_entries.append([page_id, offset, len(encoded)])

        template_entries: list[list[list[int]]] = []
        for page in store._templates:
            page_entries: list[list[int]] = []
            for template in page:
                encoded = pack_uints(template.tolist(), neuron_bits)
                offset = len(payload)
                payload.extend(encoded)
                page_entries.append([offset, len(encoded), int(template.numel())])
            template_entries.append(page_entries)

        delta_entries: list[list[list[int]]] = []
        for bank in store._delta_banks:
            page_entries = []
            for delta in bank:
                encoded = _encode_patch(delta)
                offset = len(payload)
                payload.extend(encoded)
                page_entries.append([offset, len(encoded)])
            delta_entries.append(page_entries)

        recipe_entries: list[list[int]] = []
        touched = sorted(set(store._delta_selectors) | set(store._exceptions))
        for page_instance in touched:
            encoded = bytearray()
            delta_id = store._delta_selectors.get(page_instance)
            encoded.extend(encode_uvarint(0 if delta_id is None else delta_id + 1))
            encoded.extend(_encode_patch(store._exceptions.get(page_instance, {}).items()))
            offset = len(payload)
            payload.extend(encoded)
            recipe_entries.append([page_instance, offset, len(encoded)])

        metadata = {
            "version": _VERSION,
            "vocab_size": store.vocab_size,
            "route_size": store.route_size,
            "page_size": store.page_size,
            "num_pages": store.num_pages,
            "pool_size": int(pool_size),
            "neuron_bits": neuron_bits,
            "selector_bits": selector_bits,
            "selector_defaults": [selector_default_offset, len(default_bytes)],
            "selector_overrides": selector_override_entries,
            "templates": template_entries,
            "deltas": delta_entries,
            "recipes": recipe_entries,
            "payload_crc32": zlib.crc32(payload) & 0xFFFFFFFF,
        }
        metadata_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
        temporary = destination.with_name(destination.name + ".tmp")
        with temporary.open("wb") as handle:
            handle.write(_HEADER.pack(_MAGIC, len(metadata_bytes)))
            handle.write(metadata_bytes)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        return {
            "file_bytes": destination.stat().st_size,
            "metadata_bytes": len(metadata_bytes),
            "payload_bytes": len(payload),
            "neuron_id_bits": neuron_bits,
            "selector_width_bits": selector_bits,
        }


class PackedSPRCReader:
    """Memory-mapped selective reader for a packed SPRC route container.

    Template, delta, selector-override, and recipe records are read only when the
    requested page needs them. The operating system can therefore service a page
    reconstruction with range-backed mmap faults instead of decompressing the
    complete route store.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        cache_pages: int = 256,
        verify_checksum: bool = False,
    ) -> None:
        if cache_pages < 0:
            raise ValueError("cache_pages must be non-negative")
        self.path = Path(path)
        self._file = self.path.open("rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        magic, metadata_length = _HEADER.unpack(self._mmap[: _HEADER.size])
        if magic != _MAGIC:
            self.close()
            raise ValueError("not an SPRC packed container")
        metadata_start = _HEADER.size
        metadata_end = metadata_start + metadata_length
        self.metadata = json.loads(self._mmap[metadata_start:metadata_end])
        if int(self.metadata.get("version", -1)) != _VERSION:
            self.close()
            raise ValueError("unsupported SPRC packed version")
        self._payload_offset = metadata_end
        if verify_checksum:
            checksum = zlib.crc32(self._mmap[self._payload_offset :]) & 0xFFFFFFFF
            if checksum != int(self.metadata["payload_crc32"]):
                self.close()
                raise ValueError("SPRC payload checksum mismatch")

        self.vocab_size = int(self.metadata["vocab_size"])
        self.route_size = int(self.metadata["route_size"])
        self.page_size = int(self.metadata["page_size"])
        self.num_pages = int(self.metadata["num_pages"])
        self.neuron_bits = int(self.metadata["neuron_bits"])
        self.selector_bits = int(self.metadata["selector_bits"])
        self.cache_pages = cache_pages

        self._selector_override_index = {
            int(page): (int(offset), int(length))
            for page, offset, length in self.metadata["selector_overrides"]
        }
        self._recipe_index = {
            int(instance): (int(offset), int(length))
            for instance, offset, length in self.metadata["recipes"]
        }
        self._selector_override_cache: OrderedDict[int, dict[int, int]] = OrderedDict()
        self._template_cache: OrderedDict[tuple[int, int], Tensor] = OrderedDict()
        self._delta_cache: OrderedDict[tuple[int, int], dict[int, int]] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

    def close(self) -> None:
        mmap_object = getattr(self, "_mmap", None)
        if mmap_object is not None:
            mmap_object.close()
            self._mmap = None  # type: ignore[assignment]
        file_object = getattr(self, "_file", None)
        if file_object is not None:
            file_object.close()
            self._file = None  # type: ignore[assignment]

    def __enter__(self) -> PackedSPRCReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _payload_range(self, relative_offset: int, length: int) -> tuple[int, int]:
        start = self._payload_offset + relative_offset
        return start, start + length

    def _remember(self, cache: OrderedDict, key: object, value: object) -> object:
        if not self.cache_pages:
            return value
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self.cache_pages:
            cache.popitem(last=False)
        return value

    def _selector_overrides(self, page_id: int) -> dict[int, int]:
        cached = self._selector_override_cache.get(page_id)
        if cached is not None:
            self.cache_hits += 1
            self._selector_override_cache.move_to_end(page_id)
            return cached
        indexed = self._selector_override_index.get(page_id)
        if indexed is None:
            return {}
        self.cache_misses += 1
        relative, length = indexed
        start, end = self._payload_range(relative, length)
        count, cursor = decode_uvarint(self._mmap, start)
        result: dict[int, int] = {}
        previous_token = 0
        for index in range(count):
            delta, cursor = decode_uvarint(self._mmap, cursor)
            template_id, cursor = decode_uvarint(self._mmap, cursor)
            token_id = delta if index == 0 else previous_token + delta
            result[token_id] = template_id
            previous_token = token_id
        if cursor > end:
            raise ValueError("selector override record exceeds its indexed range")
        return self._remember(self._selector_override_cache, page_id, result)  # type: ignore[return-value]

    def selector_at(self, token_id: int, page_id: int) -> int:
        if not 0 <= token_id < self.vocab_size:
            raise IndexError("token_id outside vocabulary")
        if not 0 <= page_id < self.num_pages:
            raise IndexError("page_id outside route")
        override = self._selector_overrides(page_id).get(token_id)
        if override is not None:
            return override
        relative, _ = self.metadata["selector_defaults"]
        return packed_uint_at(
            self._mmap,
            token_id,
            self.selector_bits,
            byte_offset=self._payload_offset + int(relative),
        )

    def _template(self, page_id: int, template_id: int) -> Tensor:
        key = (page_id, template_id)
        cached = self._template_cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            self._template_cache.move_to_end(key)
            return cached
        self.cache_misses += 1
        relative, length, count = self.metadata["templates"][page_id][template_id]
        start, end = self._payload_range(int(relative), int(length))
        values = unpack_uints(memoryview(self._mmap)[start:end], int(count), self.neuron_bits)
        template = torch.tensor(values, dtype=torch.long)
        return self._remember(self._template_cache, key, template)  # type: ignore[return-value]

    def _delta(self, page_id: int, delta_id: int) -> dict[int, int]:
        key = (page_id, delta_id)
        cached = self._delta_cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            self._delta_cache.move_to_end(key)
            return cached
        self.cache_misses += 1
        relative, length = self.metadata["deltas"][page_id][delta_id]
        start, end = self._payload_range(int(relative), int(length))
        patch, _ = _decode_patch(self._mmap, start, end)
        return self._remember(self._delta_cache, key, patch)  # type: ignore[return-value]

    def _recipe(self, token_id: int, page_id: int) -> tuple[int | None, dict[int, int]]:
        page_instance = token_id * self.num_pages + page_id
        indexed = self._recipe_index.get(page_instance)
        if indexed is None:
            return None, {}
        relative, length = indexed
        start, end = self._payload_range(relative, length)
        encoded_delta, cursor = decode_uvarint(self._mmap, start)
        exceptions, _ = _decode_patch(self._mmap, cursor, end)
        return (None if encoded_delta == 0 else encoded_delta - 1), exceptions

    def resolve_page(
        self,
        token_id: int,
        page_id: int,
        *,
        device: torch.device | str | None = None,
    ) -> Tensor:
        template_id = self.selector_at(token_id, page_id)
        route = self._template(page_id, template_id).clone()
        delta_id, exceptions = self._recipe(token_id, page_id)
        if delta_id is not None:
            for offset, scalar in self._delta(page_id, delta_id).items():
                route[offset] = scalar
        for offset, scalar in exceptions.items():
            route[offset] = scalar
        if device is not None:
            route = route.to(device=device, non_blocking=True)
        return route

    def resolve_slice(
        self,
        token_id: int,
        start: int,
        stop: int,
        *,
        device: torch.device | str | None = None,
    ) -> Tensor:
        if not 0 <= start <= stop <= self.route_size:
            raise IndexError("invalid route slice")
        if start == stop:
            return torch.empty(0, dtype=torch.long, device=device)
        first_page = start // self.page_size
        last_page = (stop - 1) // self.page_size
        parts: list[Tensor] = []
        for page_id in range(first_page, last_page + 1):
            page = self.resolve_page(token_id, page_id, device=device)
            page_start = page_id * self.page_size
            local_start = max(start, page_start) - page_start
            local_stop = min(stop, page_start + page.numel()) - page_start
            parts.append(page[local_start:local_stop])
        return torch.cat(parts) if len(parts) > 1 else parts[0]

    def cache_stats(self) -> dict[str, int]:
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "templates": len(self._template_cache),
            "deltas": len(self._delta_cache),
            "selector_pages": len(self._selector_override_cache),
        }
