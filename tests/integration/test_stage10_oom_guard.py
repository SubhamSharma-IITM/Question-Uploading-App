"""Regression: Stage 10 OOM guard must use anon memory, not memory.current.

Locks the post-fix behavior:
  - cgroup memory.current includes reclaimable file page cache
  - the OLD guard read memory.current and falsely tripped when file cache was
    high but anon was low — verified live root cause of the 8,338-OID
    absorption gap as of 2026-05-04
  - the NEW guard reads memory.stat and sums only non-reclaimable categories
    (anon + slab + kernel + sock + pagetables)
"""
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.pipeline.stage10_upload import UploadStage


def test_high_file_cache_does_not_trip_guard():
    """The exact production scenario: 4 GB plan, 3 GB reclaimable file cache,
    700 MB actual app anon. Old guard would say ~1 GB free; new guard says
    ~3.2 GB allocatable, which is correct."""
    memory_max = "4294967296\n"  # 4 GB
    memory_stat = (
        "anon 700000000\n"          # 668 MB actual process
        "file 3200000000\n"          # 3.05 GB page cache (reclaimable)
        "kernel 50000000\n"          # 47 MB kernel structures
        "kernel_stack 3000000\n"
        "pagetables 5000000\n"
        "sock 10000000\n"
        "vmalloc 0\n"
        "shmem 0\n"
        "active_anon 200000000\n"
        "inactive_anon 500000000\n"
        "active_file 100000000\n"
        "inactive_file 3100000000\n"
        "slab_reclaimable 100000000\n"   # NOT counted (we use 'slab' aggregate only if present)
        "slab 100000000\n"           # 95 MB
    )
    allocatable = UploadStage._parse_cgroup_anon_allocatable(memory_max, memory_stat)
    # Expected: limit - (anon + slab + kernel + sock + pagetables)
    expected_bytes = 4294967296 - (700000000 + 100000000 + 50000000 + 10000000 + 5000000)
    expected_mb = expected_bytes / (1024 * 1024)
    assert abs(allocatable - expected_mb) < 0.5, f"got {allocatable}, expected {expected_mb}"
    # And critically: it's well above the ~1.4 GB threshold for a 950 MB index.
    needed_mb = 950 * 1.5
    assert allocatable > needed_mb, (
        f"high file cache should NOT trip guard: got {allocatable:.0f} MB allocatable, "
        f"need {needed_mb} MB (this is the bug we're locking out)"
    )


def test_high_anon_correctly_trips_guard():
    """Real memory pressure: anon is high, file cache is low. Guard must trip."""
    memory_max = "4294967296\n"  # 4 GB
    memory_stat = (
        "anon 3500000000\n"          # 3.34 GB anon — real pressure
        "file 100000000\n"           # 95 MB cache
        "kernel 50000000\n"
        "pagetables 5000000\n"
        "sock 10000000\n"
        "slab 100000000\n"
    )
    allocatable = UploadStage._parse_cgroup_anon_allocatable(memory_max, memory_stat)
    # Expected: 4 GB - (3.5 + 0.1 + 0.05 + 0.005 + 0.01) GB = ~635 MB
    needed_mb = 950 * 1.5  # ~1425 MB
    assert allocatable < needed_mb, (
        f"high anon SHOULD trip guard: got {allocatable:.0f} MB allocatable, "
        f"need {needed_mb} MB (real OOM scenario)"
    )


def test_unbounded_limit_returns_none():
    """Some hosts report 'max' for unbounded — no guard needed."""
    assert UploadStage._parse_cgroup_anon_allocatable("max\n", "anon 1000\n") is None
    assert UploadStage._parse_cgroup_anon_allocatable("\n", "anon 1000\n") is None


def test_malformed_inputs_handled():
    """Non-numeric limit or empty stat — return None or 0-ish, never crash."""
    # Garbage limit
    assert UploadStage._parse_cgroup_anon_allocatable("notanumber", "anon 100") is None
    # Empty stat: limit minus 0 = full limit available
    out = UploadStage._parse_cgroup_anon_allocatable("4294967296", "")
    assert out is not None and abs(out - 4096) < 1


def test_partial_stat_uses_zero_default():
    """If memory.stat is missing some keys we expect, the missing ones default to 0."""
    memory_stat = "anon 100000000\n"  # only anon present
    out = UploadStage._parse_cgroup_anon_allocatable("4294967296", memory_stat)
    expected = (4294967296 - 100000000) / (1024 * 1024)
    assert abs(out - expected) < 0.5


def test_slab_not_double_counted_with_slab_reclaimable():
    """We sum 'slab' only — 'slab_reclaimable' and 'slab_unreclaimable' are
    components of 'slab', not separate buckets. Real Render output has both;
    our parser must use only the aggregate to avoid double-counting."""
    memory_stat = (
        "anon 1000000000\n"
        "slab 100000000\n"
        "slab_reclaimable 80000000\n"      # should NOT be added
        "slab_unreclaimable 20000000\n"     # should NOT be added
        "kernel 50000000\n"
    )
    out = UploadStage._parse_cgroup_anon_allocatable("4294967296", memory_stat)
    # Expected: 4 GB - (1.0 + 0.1 + 0.05) GB = ~2.81 GB
    expected_bytes = 4294967296 - (1000000000 + 100000000 + 50000000)
    expected_mb = expected_bytes / (1024 * 1024)
    assert abs(out - expected_mb) < 0.5, f"got {out}, expected {expected_mb} (no double-counting)"


def test_real_render_snapshot_2026_05_04():
    """Use the actual numbers we read live from Render at the moment of bug
    discovery. Locks: this exact production state must NOT trip the guard."""
    # Verified 2026-05-04 right before back-fill via SSH cat /sys/fs/cgroup/memory.stat
    memory_max = "4294967296"  # 4 GB
    memory_stat = (
        "anon 701952000\n"
        "file 3330822144\n"
        "kernel 129269760\n"
        "kernel_stack 3309568\n"
        "pagetables 3174400\n"
        "sec_pagetables 0\n"
        "percpu 1728\n"
        "sock 24576\n"
        "vmalloc 0\n"
        "shmem 0\n"
        "file_mapped 21176320\n"
        "file_dirty 32575488\n"
        "active_anon 237920256\n"
        "inactive_anon 463978496\n"
        "active_file 135315456\n"
        "inactive_file 3195506688\n"
        "slab_reclaimable 118292264\n"
    )
    allocatable = UploadStage._parse_cgroup_anon_allocatable(memory_max, memory_stat)
    # No 'slab' aggregate in this slice (only slab_reclaimable). Anon ~669 MB,
    # kernel ~123 MB, sock ~0.02 MB, pagetables ~3 MB → non_reclaimable ~795 MB.
    # Allocatable ~ 4096 - 795 = ~3301 MB.
    assert allocatable > 3000, (
        f"Live-Render snapshot should yield ~3 GB allocatable, got {allocatable:.0f} MB. "
        f"This is the smoking-gun: under the OLD guard, memory.current=2.84 GB "
        f"would have shown only 1.16 GB free."
    )
    # And critically: well above the threshold for a 1 GB index
    needed_mb = 1000 * 1.5
    assert allocatable > needed_mb, (
        f"Live-Render snapshot must allow absorption of a 1 GB index: "
        f"got {allocatable:.0f} MB, need {needed_mb} MB"
    )


if __name__ == "__main__":
    test_high_file_cache_does_not_trip_guard()
    test_high_anon_correctly_trips_guard()
    test_unbounded_limit_returns_none()
    test_malformed_inputs_handled()
    test_partial_stat_uses_zero_default()
    test_slab_not_double_counted_with_slab_reclaimable()
    test_real_render_snapshot_2026_05_04()
    print("OK — 7 tests passed")
