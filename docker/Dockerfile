# ISLES 2026 SADL - Grand Challenge "invoke" API implementation.
#
# 11-member nnU-Net ensemble with GPU-side plans->native inverse resample and
# Pillar-1 policy fusion. Checkpoints and policy JSON are provided at run time
# via /opt/ml/model/ (see manifest_pillar1_v2.json for the expected layout:
# checkpoints/<experiment>/fold_{0..4}/<checkpoint_name> + plans/dataset
# JSONs, plus policy/policy.json referenced from the manifest).
#
# NOTE: The trained model bundle (nnU-Net checkpoints + policy JSON + the
# manifest_pillar1_v2.json in this directory) is NOT baked into this image.
# Provide it at container run time by bind-mounting the extracted model tree
# at /opt/ml/model/.
#
# Build with (from project root, NOT from docker/):
#   docker build --platform=linux/amd64 --tag isles2026_sadl -f docker/Dockerfile .
#
# The wrapper script docker/do_build.sh does exactly that.

FROM --platform=linux/amd64 pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime AS isles2026_sadl

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    nnUNet_raw=/tmp/nnunet/raw \
    nnUNet_preprocessed=/tmp/nnunet/preprocessed \
    nnUNet_results=/tmp/nnunet/results \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=2 \
    DEBIAN_FRONTEND=noninteractive

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN git clone --depth 1 --branch v0.2.6 \
        https://github.com/MIC-DKFZ/acvl_utils.git /tmp/acvl_utils_repo \
    && mkdir -p /opt/app \
    && cp -r /tmp/acvl_utils_repo/acvl_utils /opt/app/acvl_utils \
    && rm -rf /tmp/acvl_utils_repo \
    && chown -R user:user /opt/app

RUN mkdir -p /tmp/nnunet/raw /tmp/nnunet/preprocessed /tmp/nnunet/results \
    && chown -R user:user /tmp/nnunet

USER user
WORKDIR /opt/app

RUN python -m venv --system-site-packages --without-pip /home/user/venv
ENV PATH="/home/user/venv/bin:$PATH"

COPY --chown=user:user docker/requirements.txt /opt/app/requirements.txt
RUN python -m pip install \
        --no-cache-dir \
        --no-color \
        --requirement /opt/app/requirements.txt

COPY --chown=user:user third_party/nnUNet /opt/nnUNet
RUN python -m pip install --no-cache-dir --no-deps -e /opt/nnUNet

COPY --chown=user:user Code/src/nnunet_isles /opt/app/nnunet_isles
ENV PYTHONPATH=/opt/app:${PYTHONPATH}

# Server + inference adapter (GPU-side inverse resample).
COPY --chown=user:user docker/app.py /opt/app/app.py
COPY --chown=user:user docker/inference.py /opt/app/inference.py

LABEL org.grand-challenge.api-method="invoke"
LABEL sadl.build.version="1.0.0"
EXPOSE 4743
ENTRYPOINT ["python", "app.py"]
