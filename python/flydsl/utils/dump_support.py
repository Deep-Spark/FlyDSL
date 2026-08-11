# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Whether official IR/ISA dump helpers are compiled into this install.

Source checkouts keep ``DUMP_SUPPORT = True``. Release wheels built with
``FLYDSL_RELEASE_STRIP_DUMP=1`` rewrite this file to ``False`` during packaging
so dump helpers cannot be revived by runtime monkeypatching.
"""

DUMP_SUPPORT = True


def require_dump_support(*, feature: str) -> None:
    """Raise if official dump helpers were stripped from this build."""
    if not DUMP_SUPPORT:
        raise RuntimeError(
            f"{feature} is stripped from this FlyDSL build (DUMP_SUPPORT=False). "
            "Official IR and ISA dump helpers are not available in release artifacts."
        )
