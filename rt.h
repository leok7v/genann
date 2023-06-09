#pragma once
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// tiny Windows runtime for missing stuff

#ifdef cplusplus
extern "C" {
#endif

#define thread_local __declspec(thread)

#define println(...) printline(__FILE__, __LINE__, __func__, "" __VA_ARGS__)

#define assertion(b, ...) do {                                              \
    if (!(b)) {                                                             \
        println("%s false\n", #b); println("" __VA_ARGS__);                 \
        printf("%s false\n", #b); printf("" __VA_ARGS__); printf("\n");     \
        __debugbreak();                                                     \
        exit(1);                                                            \
    }                                                                       \
} while (0) // better assert

#define static_assertion(b) static_assert(b, #b)

#ifndef countof
    #define countof(a) (sizeof(a) / sizeof((a)[0])) // MS is _countof()
#endif

#define null ((void*)0) // like nullptr but for C99

#if defined(__GNUC__) || defined(__clang__)
#define attribute_packed __attribute__((packed))
#define begin_packed
#define end_packed attribute_packed
#else
#define begin_packed __pragma( pack(push, 1) )
#define end_packed __pragma( pack(pop) )
#define attribute_packed !!! use begin_packed/end_packed instead !!!
#endif

// usage: typedef begin_packed struct foo_s { ... } end_packed foo_t;

enum {
    NSEC_IN_USEC = 1000,
    NSEC_IN_MSEC = NSEC_IN_USEC * 1000,
    NSEC_IN_SEC  = NSEC_IN_MSEC * 1000
};

uint32_t random32(uint32_t* state); // state aka seed
double   seconds();     // seconds since boot (3/10MHz resolution)
int64_t  nanoseconds(); // nanoseconds since boot (3/10MHz resolution)
void     printline(const char* file, int line, const char* func,
                   const char* format, ...);


#define fatal_if(b, ...) do {                  \
    bool _b_ = (b);                       \
    if (_b_) {                            \
        printline(__FILE__, __LINE__, __func__, "" __VA_ARGS__); \
        fprintf(stderr, "%s failed", #b); \
        exit(1);                          \
    }                                     \
} while (0)

#ifdef RT_IMPLEMENTATION

uint32_t random32(uint32_t* state) {
    // https://gist.github.com/tommyettinger/46a874533244883189143505d203312c
    static thread_local int init; // first seed must be odd
    if (init == 0) { init = 1; *state |= 1; }
    uint32_t z = (*state += 0x6D2B79F5UL);
    z = (z ^ (z >> 15)) * (z | 1UL);
    z ^= z + (z ^ (z >> 7)) * (z | 61UL);
    return z ^ (z >> 14);
}

typedef union LARGE_INTEGER {
    struct {
        uint32_t LowPart;
        int32_t HighPart;
    };
    int64_t QuadPart;
} LARGE_INTEGER;

#pragma comment(lib, "kernel32")

int32_t __stdcall QueryPerformanceCounter(LARGE_INTEGER* lpPerformanceCount);
int32_t __stdcall QueryPerformanceFrequency(LARGE_INTEGER* lpFrequency);
void    __stdcall OutputDebugStringA(const char* s);

#define std_input_handle    ((uint32_t)-10)
#define std_output_handle   ((uint32_t)-11)
#define std_error_handle    ((uint32_t)-12)

void*   __stdcall GetStdHandle(uint32_t stdhandle);


double seconds() { // since_boot
    LARGE_INTEGER qpc;
    QueryPerformanceCounter(&qpc);
    static double one_over_freq;
    if (one_over_freq == 0) {
        LARGE_INTEGER frequency;
        QueryPerformanceFrequency(&frequency);
        one_over_freq = 1.0 / frequency.QuadPart;
    }
    return (double)qpc.QuadPart * one_over_freq;
}

int64_t nanoseconds() {
    return (int64_t)(seconds() * NSEC_IN_SEC);
}

void printline(const char* file, int line, const char* func,
        const char* format, ...) {
    char text[1024];
    va_list vl;
    va_start(vl, format);
    char* p = text + snprintf(text, countof(text), "%s(%d): %s ", file, line, func);
    vsnprintf(p, countof(text) - (p - text), format, vl);
    text[countof(text) - 1] = 0;
    text[countof(text) - 2] = 0;
    size_t n = strlen(text);
    if (n > 0 && text[n - 1] != '\n') { text[n] = '\n'; text[n + 1] = 0;  }
    va_end(vl);
    OutputDebugStringA(text);
    fprintf(stderr, "%s", text);
}

#endif // RT_IMPLEMENTATION

#ifdef cplusplus
} // extern "C"
#endif
