FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    WANDB_MODE=offline

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# requirements.txt pins jax[cuda12_local], which assumes a system CUDA
# toolkit -- true on the conda host env, not inside this container. Force
# cuda12's self-contained pip wheels instead, for the container build only,
# with exact nvidia-*-cu12 versions pinned so a future rebuild can't
# silently drift past a worker node driver's max supported CUDA runtime.
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir --force-reinstall \
       "jax[cuda12]==0.10.2" \
       "nvidia-cublas-cu12==12.9.2.10" \
       "nvidia-cuda-cupti-cu12==12.9.79" \
       "nvidia-cuda-nvcc-cu12==12.9.86" \
       "nvidia-cuda-nvrtc-cu12==12.9.86" \
       "nvidia-cuda-runtime-cu12==12.9.79" \
       "nvidia-cudnn-cu12==9.26.0.51" \
       "nvidia-cufft-cu12==11.4.1.4" \
       "nvidia-cusolver-cu12==11.7.5.82" \
       "nvidia-cusparse-cu12==12.5.10.65" \
       "nvidia-nccl-cu12==2.31.2" \
       "nvidia-nvjitlink-cu12==12.9.86" \
       "nvidia-nvshmem-cu12==3.7.2"

# Source subset needed to run the PennyLane/JAX/GPU smoke tests below -- not
# the full pipeline. Real experiments still mount the live checkout over
# /workspace (see docker/README.md); this is only for verifying the installed
# dependency stack works before that.
COPY case_reader.py ./
COPY helpers/utils.py helpers/config.py helpers/trained_run.py ./helpers/
COPY quantum/architectures.py quantum/losses.py quantum/math_functions.py ./quantum/
COPY quantum/circuits/base.py quantum/circuits/vqc.py quantum/circuits/registry.py ./quantum/circuits/
COPY tests/test_circuit_parity.py tests/test_end_to_end.py ./tests/
COPY docker/tests/make_smoke_data.py docker/tests/check_gpu.py ./docker/tests/

# Tiny synthetic JetClass-shaped fixtures (100 signal + 100 background jets)
# for tests.test_end_to_end.TestBalancedJetClassLoader, which otherwise needs
# a real JetClass mount. JETCLASS_DIR overrides that test's default path
# without touching host behaviour (unset there, so it keeps using
# /ceph/abal/JetClass).
RUN python docker/tests/make_smoke_data.py --out-dir /opt/smoke_data/JetClass
ENV JETCLASS_DIR=/opt/smoke_data/JetClass

CMD ["sh", "-c", "python docker/tests/check_gpu.py && python -m unittest -v \
     tests.test_circuit_parity \
     tests.test_end_to_end.TestJaxFullTrainingLoop \
     tests.test_end_to_end.TestBalancedJetClassLoader"]
