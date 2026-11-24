#!/usr/bin/env bash
# Downloads ARM astcenc 3.5 and builds the exact game DXT5 encoder binaries
# (Intel ISPC Texture Compressor SSE2 BC3 + stb_dxt single-colour block encoder)
# into ./bin/. Needed for Stage 2 refinement and training; skip if only running
# Stage 1 with `clean.py --no-refiner`.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BIN="$ROOT/bin"
BUILD="$ROOT/native/_build"
mkdir -p "$BIN" "$BUILD"

# Pick C++ compiler (system g++/clang++, or ziglang via Python as fallback)
if command -v g++ >/dev/null 2>&1; then
    CXX=(g++)
elif command -v clang++ >/dev/null 2>&1; then
    CXX=(clang++)
elif python3 -c "import ziglang" >/dev/null 2>&1; then
    CXX=(python3 -m ziglang c++)
else
    echo "Error: no C++ compiler found. Install g++/clang++ or run 'pip install ziglang'." >&2
    exit 1
fi
echo "Using C++ compiler: ${CXX[*]}"

# 1. ARM astcenc 3.5 (Linux x86-64)
if [ ! -x "$BIN/astcenc-3.5" ]; then
    echo "[1/3] Downloading ARM astcenc 3.5..."
    curl -sL "https://github.com/ARM-software/astc-encoder/releases/download/3.5/astcenc-3.5-linux-x64.zip" -o "$BUILD/astc35.zip"
    rm -rf "$BUILD/astc35"
    python3 -c "import zipfile; zipfile.ZipFile('$BUILD/astc35.zip').extractall('$BUILD/astc35')"
    if grep -qm1 avx2 /proc/cpuinfo 2>/dev/null && [ -f "$BUILD/astc35/astcenc/astcenc-avx2" ]; then
        cp "$BUILD/astc35/astcenc/astcenc-avx2" "$BIN/astcenc-3.5"
    else
        cp "$BUILD/astc35/astcenc/astcenc-sse4.1" "$BIN/astcenc-3.5"
    fi
    chmod +x "$BIN/astcenc-3.5"
    echo "  -> $BIN/astcenc-3.5"
fi

# 2. stb_dxt single-colour BC3 encoder
if [ ! -x "$BIN/bin_enc_stb" ]; then
    echo "[2/3] Building bin_enc_stb..."
    curl -sL "https://raw.githubusercontent.com/nothings/stb/master/stb_dxt.h" -o "$BUILD/stb_dxt.h"
    "${CXX[@]}" -O2 -w -I"$BUILD" "$ROOT/native/enc_stb.cpp" -o "$BIN/bin_enc_stb"
    chmod +x "$BIN/bin_enc_stb"
    echo "  -> $BIN/bin_enc_stb"
fi

# 3. Intel ISPC Texture Compressor (BC3, SSE2 precise math)
if [ ! -x "$BIN/bin_ispcv_head_precise_sse2" ]; then
    echo "[3/3] Building bin_ispcv_head_precise_sse2..."
    if ! command -v ispc >/dev/null 2>&1; then
        ISPC_VER="1.26.0"
        if [ ! -x "$BUILD/ispc/bin/ispc" ]; then
            curl -sL "https://github.com/ispc/ispc/releases/download/v${ISPC_VER}/ispc-v${ISPC_VER}-linux.tar.gz" -o "$BUILD/ispc.tar.gz"
            mkdir -p "$BUILD/ispc"
            tar -xzf "$BUILD/ispc.tar.gz" -C "$BUILD/ispc" --strip-components=1
        fi
        ISPC="$BUILD/ispc/bin/ispc"
    else
        ISPC="$(command -v ispc)"
    fi
    if [ ! -d "$BUILD/ISPCTextureCompressor" ]; then
        git clone --quiet "https://github.com/GameTechDev/ISPCTextureCompressor.git" "$BUILD/ISPCTextureCompressor"
    fi
    REV="79ddbc90334fc31edd438e68ccb0fe99b4e15aab"
    KDIR="$BUILD/ispc_src"
    mkdir -p "$KDIR"
    for f in kernel.ispc ispc_texcomp.cpp ispc_texcomp.h; do
        git -C "$BUILD/ISPCTextureCompressor" show "${REV}:ispc_texcomp/${f}" > "$KDIR/$f"
    done
    "$ISPC" -O2 --woff --arch=x86-64 --target=sse2-i32x4 --pic --opt=disable-fma \
        -o "$KDIR/kernel.o" -h "$KDIR/kernel_ispc.h" "$KDIR/kernel.ispc"
    "${CXX[@]}" -O2 -w -I"$KDIR" "$ROOT/native/enc_ispc.cpp" "$KDIR/ispc_texcomp.cpp" "$KDIR/kernel.o" \
        -o "$BIN/bin_ispcv_head_precise_sse2"
    chmod +x "$BIN/bin_ispcv_head_precise_sse2"
    echo "  -> $BIN/bin_ispcv_head_precise_sse2"
fi

echo "All encoder binaries ready in $BIN:"
ls -lh "$BIN"
