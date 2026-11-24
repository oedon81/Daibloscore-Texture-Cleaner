// enc_ispc <in.rgba> <w> <h> <out.bc3>   Intel ISPC Texture Compressor, BC3
#include "ispc_texcomp.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    int w = atoi(argv[2]), h = atoi(argv[3]);
    std::vector<unsigned char> img(w * h * 4), out(w * h);
    FILE* f = fopen(argv[1], "rb");
    if (!f || fread(img.data(), 1, img.size(), f) != img.size()) return 1;
    fclose(f);
    rgba_surface s;
    s.ptr = img.data();
    s.width = w;
    s.height = h;
    s.stride = w * 4;
    CompressBlocksBC3(&s, out.data());
    f = fopen(argv[4], "wb");
    if (!f || fwrite(out.data(), 1, out.size(), f) != out.size()) return 1;
    fclose(f);
    return 0;
}
