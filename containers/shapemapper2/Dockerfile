# syntax=docker/dockerfile:1.7

# Public candidate: compile the unchanged upstream v2.3 source tag and use
# source-traceable Debian packages for runtime dependencies.
FROM --platform=linux/amd64 debian:bookworm-slim@sha256:362e64223cc0da95422b3b13c045186fc0a81250e765d31c025fbddf257f6143 AS build

ARG SHAPEMAPPER_VERSION=2.3
ARG SHAPEMAPPER_SOURCE_SHA256=1c1fb1e6e03249715eedb822217c8c3b5d18a0d60724ca4d403d88f29f866b6f
ARG SHAPEMAPPER_SOURCE_URL=https://codeload.github.com/Weeks-UNC/shapemapper2/tar.gz/refs/tags/v2.3

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        g++ \
        libboost-filesystem-dev \
        libboost-iostreams-dev \
        libboost-program-options-dev \
        libboost-system-dev \
        make \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

ADD --checksum=sha256:1c1fb1e6e03249715eedb822217c8c3b5d18a0d60724ca4d403d88f29f866b6f \
    https://codeload.github.com/Weeks-UNC/shapemapper2/tar.gz/refs/tags/v2.3 \
    /tmp/shapemapper2-v2.3.tar.gz

RUN test "${SHAPEMAPPER_VERSION}" = "2.3" \
    && test "${SHAPEMAPPER_SOURCE_SHA256}" = "1c1fb1e6e03249715eedb822217c8c3b5d18a0d60724ca4d403d88f29f866b6f" \
    && test "${SHAPEMAPPER_SOURCE_URL}" = "https://codeload.github.com/Weeks-UNC/shapemapper2/tar.gz/refs/tags/v2.3" \
    && tar -xzf /tmp/shapemapper2-v2.3.tar.gz -C /opt \
    && mv /opt/shapemapper2-2.3 /opt/shapemapper2 \
    && rm /tmp/shapemapper2-v2.3.tar.gz \
    && cmake -S /opt/shapemapper2 -B /tmp/shapemapper-build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/shapemapper-build --parallel 2

FROM --platform=linux/amd64 debian:bookworm-slim@sha256:362e64223cc0da95422b3b13c045186fc0a81250e765d31c025fbddf257f6143

ARG SHAPEMAPPER_VERSION=2.3
ARG SHAPEMAPPER_SOURCE_SHA256=1c1fb1e6e03249715eedb822217c8c3b5d18a0d60724ca4d403d88f29f866b6f

LABEL org.opencontainers.image.title="NERD ShapeMapper2" \
      org.opencontainers.image.description="ShapeMapper2 v2.3 built from the unchanged upstream source tag with Debian runtime packages" \
      org.opencontainers.image.source="https://github.com/LucksLab/nerd" \
      org.opencontainers.image.url="https://github.com/LucksLab/nerd" \
      org.opencontainers.image.version="2.3-r0-candidate" \
      org.opencontainers.image.vendor="Lucks Lab" \
      org.opencontainers.image.documentation="https://github.com/LucksLab/nerd/tree/main/containers/shapemapper2" \
      io.nerd.shapemapper.upstream="https://github.com/Weeks-UNC/shapemapper2/tree/v2.3" \
      io.nerd.shapemapper.source-sha256="1c1fb1e6e03249715eedb822217c8c3b5d18a0d60724ca4d403d88f29f866b6f" \
      io.nerd.shapemapper.distribution="public-candidate"

# Keep /usr/share/doc: Debian copyright files are part of the redistribution
# record and intentionally are not removed as a size optimization.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bbmap \
        bowtie2 \
        ca-certificates \
        fonts-dejavu-core \
        ghostscript \
        graphviz \
        libboost-filesystem1.74.0 \
        libboost-iostreams1.74.0 \
        libboost-program-options1.74.0 \
        libboost-system1.74.0 \
        pv \
        python3 \
        python3-matplotlib \
        python3-numpy \
        python3-scipy \
        python3-sklearn \
        rna-star \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/shapemapper2 /opt/shapemapper2
COPY LICENSES/ShapeMapper-MIT.txt /usr/share/licenses/nerd-shapemapper2/ShapeMapper-MIT.txt
COPY PUBLIC_THIRD_PARTY_NOTICES.md /usr/share/licenses/nerd-shapemapper2/THIRD_PARTY_NOTICES.md

ENV SHAPEMAPPER_ROOT_DIR=/opt/shapemapper2 \
    PATH=/opt/shapemapper2:/opt/shapemapper2/internals/bin:/usr/share/bbmap:${PATH}

WORKDIR /work

ENTRYPOINT ["/opt/shapemapper2/shapemapper"]
CMD ["--help"]
