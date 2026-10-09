// A CPython entry point linked with the MemorySanitizer runtime. MSan has no
// shared runtime that could be preloaded, so it must live in the executable.
#include <Python.h>

int main(int argc, char **argv) { return Py_BytesMain(argc, argv); }
