"""
Explicit GPU-visibility check for JAX inside the container. Guards against
the silent-CPU-fallback risk documented in llm_summary.MD's JAX backend
section (jax.devices() reporting no CudaDevice while everything else still
appears to run) -- fails loudly instead of letting the unit tests below pass
on CPU without anyone noticing.

Author: Aritra Bal (ETP)
Date: 2026-09-13
"""
import jax

devices = jax.devices()
print("jax.devices():", devices)
platforms = {d.platform for d in devices}
if "gpu" not in platforms:
    raise SystemExit(f"No GPU-backed JAX device found (platforms seen: {platforms})")
print("GPU check passed.")
