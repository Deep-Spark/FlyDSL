# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar MR async-copy synchronization helpers (commit / wait)."""

from ..._mlir.dialects import fly_ixdl


def cp_async_commit_group(*, loc=None, ip=None):
    """Seal outstanding ``copy_atom_call`` async issues into the current group."""
    return fly_ixdl.CpAsyncCommitGroupOp(loc=loc, ip=ip)


def cp_async_wait_group(n=0, *, loc=None, ip=None):
    """Wait until async copy group ``n`` is complete."""
    return fly_ixdl.CpAsyncWaitGroupOp(n=int(n), loc=loc, ip=ip)
