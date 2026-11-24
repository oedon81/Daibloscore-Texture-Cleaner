// enc_stb <in.rgba> <w> <h> <out.bc3> <mode>   (mode: 0 normal, 2 high quality)
#define STB_DXT_IMPLEMENTATION
#include "stb_dxt.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    int w = atoi(argv[2]), h = atoi(argv[3]), mode = atoi(argv[5]);
    std::vector<unsigned char> img(w * h * 4), out(w * h);
    FILE* f = fopen(argv[1], "rb");
    if (!f || fread(img.data(), 1, img.size(), f) != img.size()) return 1;
    fclose(f);
    unsigned char* o = out.data();
    for (int y = 0; y < h; y += 4) {
        for (int x = 0; x < w; x += 4, o += 16) {
            unsigned char b[64];
            for (int j = 0; j < 4; j++)
                for (int i = 0; i < 4; i++)
                    for (int c = 0; c < 4; c++)
                        b[(j * 4 + i) * 4 + c] = img[((y + j) * w + x + i) * 4 + c];
            stb_compress_dxt_block(o, b, 1, mode);
        }
    }
    f = fopen(argv[4], "wb");
    if (!f || fwrite(out.data(), 1, out.size(), f) != out.size()) return 1;
    fclose(f);
    return 0;
}
