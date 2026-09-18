from functools import cache
from pathlib import Path

from torch.utils.cpp_extension import load


@cache
def load_extension() -> None:
    load(
        name="omniflow_flux_hipblaslt_fixed",
        sources=[str(Path(__file__).with_suffix(".cpp"))],
        extra_include_paths=["/opt/rocm/include"],
        extra_cflags=["-O3"],
        extra_ldflags=["-L/opt/rocm/lib", "-lhipblaslt", "-ltorch_hip"],
        is_python_module=False,
        verbose=False,
    )
