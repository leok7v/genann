#define RT_IMPLEMENTATION
#include "rt.h"

int main1(int argc, const char* argv[]); // extern see main1.c
int main2(int argc, const char* argv[]);
int main3(int argc, const char* argv[]);

int main(int argc, const char* argv[]) {
    int c = argc > 1 ? argv[1][0] - '0' : 0;
    switch (c) {
        case 1: return main1(argc, argv);
        case 2: return main2(argc, argv);
        case 3: return main3(argc, argv);
        default: return 1;
    }
}
