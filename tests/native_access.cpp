#include <cstddef>
#include <cstdint>

extern "C" void unchecked_native_store(int64_t *values, size_t index) {
  values[index] = 123;
}
