#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <sanitizer/asan_interface.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <cstdio>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

using Destructor = void (*)(void *, size_t, void *);
struct ExternalAllocator {
  void *(*malloc)(size_t, void *);
  void *(*realloc)(void *, size_t, void *);
  void (*free)(void *, void *);
  void *opaque;
};

// Private Numba 0.62.1 ABI; version checked by the Python integration.
struct MemInfo {
  std::atomic_size_t refcount;
  Destructor destructor;
  void *info;
  void *data;
  size_t size;
  ExternalAllocator *external;
};
static_assert(sizeof(MemInfo) == 48, "Unsupported NRT MemInfo ABI");

using AllocateMetadata = MemInfo *(*)(size_t, Destructor);
static AllocateMetadata allocate_metadata;

struct Payload {
  void *base;
  size_t allocation_size;
  ExternalAllocator *external;
  Destructor user_destructor;
  bool safe;
};

static void destroy_payload(void *data, size_t size, void *info) {
  auto *payload = static_cast<Payload *>(info);
  if (payload->user_destructor)
    payload->user_destructor(data, size, nullptr);
  if (payload->safe)
    memset(data, 0xDE, size);
  if (payload->external) {
    // A custom allocator may write free-list metadata inside the returned block.
    __asan_unpoison_memory_region(payload->base, payload->allocation_size);
    payload->external->free(payload->base, payload->external->opaque);
  } else {
    free(payload->base);
  }
  free(payload);
}

static MemInfo *allocate_payload(size_t size, unsigned alignment, bool safe,
                                 Destructor destructor,
                                 ExternalAllocator *external) {
  if (!alignment || (alignment & (alignment - 1))) {
    errno = EINVAL;
    return nullptr;
  }
  size_t align = alignment < sizeof(void *) ? sizeof(void *) : alignment;
  auto *payload = static_cast<Payload *>(malloc(sizeof(Payload)));
  if (!payload)
    return nullptr;
  void *data = nullptr;
  void *base = nullptr;
  size_t allocation_size = size;
  if (external) {
    constexpr size_t guard = 64;
    if (!external->malloc || !external->free ||
        align > std::numeric_limits<size_t>::max() - 2 * guard ||
        size > std::numeric_limits<size_t>::max() - align - 2 * guard) {
      free(payload);
      errno = EINVAL;
      return nullptr;
    }
    allocation_size = size + align + 2 * guard;
    base = external->malloc(allocation_size, external->opaque);
    if (base) {
      uintptr_t start = reinterpret_cast<uintptr_t>(base);
      uintptr_t aligned = (start + guard + align - 1) & ~(align - 1);
      data = reinterpret_cast<void *>(aligned);
      __asan_unpoison_memory_region(base, allocation_size);
      __asan_poison_memory_region(base, allocation_size);
      __asan_unpoison_memory_region(data, size);
    }
  } else {
    int error = posix_memalign(&base, align, size);
    if (error)
      errno = error;
    data = base;
  }
  if (!base) {
    free(payload);
    return nullptr;
  }
  *payload = {base, allocation_size, external, destructor, safe};
  MemInfo *metadata = allocate_metadata(0, nullptr);
  if (!metadata) {
    if (external) {
      __asan_unpoison_memory_region(base, allocation_size);
      external->free(base, external->opaque);
    } else {
      free(base);
    }
    free(payload);
    return nullptr;
  }
  metadata->data = data;
  metadata->size = size;
  metadata->info = payload;
  metadata->destructor = destroy_payload;
  if (safe)
    memset(data, 0xCB, size);
  return metadata;
}

extern "C" void numba_asan_init_nrt(void *allocate) {
  allocate_metadata = reinterpret_cast<AllocateMetadata>(allocate);
}
extern "C" MemInfo *numba_asan_alloc(size_t size) {
  return allocate_payload(size, 64, false, nullptr, nullptr);
}
extern "C" MemInfo *numba_asan_alloc_safe(size_t size) {
  return allocate_payload(size, 64, true, nullptr, nullptr);
}
extern "C" MemInfo *numba_asan_alloc_aligned(size_t size, unsigned alignment) {
  return allocate_payload(size, alignment, false, nullptr, nullptr);
}
extern "C" MemInfo *numba_asan_alloc_safe_aligned(size_t size, unsigned alignment) {
  return allocate_payload(size, alignment, true, nullptr, nullptr);
}
extern "C" MemInfo *numba_asan_alloc_dtor(size_t size, Destructor destructor) {
  return allocate_payload(size, 64, false, destructor, nullptr);
}
extern "C" MemInfo *numba_asan_alloc_dtor_safe(size_t size, Destructor destructor) {
  return allocate_payload(size, 64, true, destructor, nullptr);
}
extern "C" MemInfo *numba_asan_alloc_safe_aligned_external(
    size_t size, unsigned alignment, ExternalAllocator *external) {
  return allocate_payload(size, alignment, true, nullptr, external);
}

struct RuntimeAPI {
  MemInfo *(*allocate)(size_t);
  MemInfo *(*allocate_external)(size_t, ExternalAllocator *);
  MemInfo *(*manage_memory)(void *, void (*)(void *));
  void (*acquire)(MemInfo *);
  void (*release)(MemInfo *);
  void *(*get_data)(MemInfo *);
};
static RuntimeAPI runtime_api;
static MemInfo *allocate_external(size_t size, ExternalAllocator *external) {
  return allocate_payload(size, 64, false, nullptr, external);
}
extern "C" void numba_asan_init_api(const RuntimeAPI *original) {
  runtime_api = *original;
  runtime_api.allocate = numba_asan_alloc;
  runtime_api.allocate_external = allocate_external;
}
extern "C" const RuntimeAPI *numba_asan_get_api() { return &runtime_api; }

static void *numpy_malloc(void *, size_t size) { return malloc(size); }
static void *numpy_calloc(void *, size_t count, size_t size) {
  return calloc(count, size);
}
static void *numpy_realloc(void *, void *ptr, size_t size) {
  return realloc(ptr, size);
}
static void numpy_free(void *, void *ptr, size_t) { free(ptr); }

extern "C" int numba_asan_install_numpy() {
  if (_import_array() < 0)
    return -1;
  static PyDataMem_Handler handler = {
      "numba_asan_malloc", 1,
      {nullptr, numpy_malloc, numpy_calloc, numpy_realloc, numpy_free}};
  PyObject *capsule = PyCapsule_New(&handler, "mem_handler", nullptr);
  if (!capsule)
    return -1;
  PyObject *previous = PyDataMem_SetHandler(capsule);
  Py_DECREF(capsule);
  if (!previous)
    return -1;
  Py_DECREF(previous);
  return 0;
}

struct Frame {
  uintptr_t address;
  uintptr_t size;
  uintptr_t object_offset;
  char *name;
  char *object;
  char *symbolizer;
  Frame *next;
};
static std::atomic<Frame *> frames{nullptr};

extern "C" void numba_asan_register_frame(uintptr_t address, uintptr_t size,
                                         uintptr_t offset, const char *name,
                                         const char *object,
                                         const char *symbolizer) {
  auto *frame = static_cast<Frame *>(malloc(sizeof(Frame)));
  if (!frame)
    abort();
  *frame = {address, size, offset, strdup(name), strdup(object),
            strdup(symbolizer), nullptr};
  if (!frame->name || !frame->object || !frame->symbolizer)
    abort();
  frame->next = frames.load();
  while (!frames.compare_exchange_weak(frame->next, frame)) {}
}

static void report_jit_frames(const char *report) {
  const char *line = report;
  while ((line = strchr(line, '#'))) {
    unsigned number;
    unsigned long pc;
    if (sscanf(line, "#%u 0x%lx", &number, &pc) == 2) {
      for (Frame *frame = frames.load(); frame; frame = frame->next) {
        if (pc < frame->address || pc - frame->address >= frame->size)
          continue;
        uintptr_t offset = frame->object_offset + pc - frame->address;
        fprintf(stderr, "\nNumba ASan JIT frame #%u: %s\n", number, frame->name);
        fprintf(stderr, "  object: %s, offset: 0x%lx\n", frame->object,
                static_cast<unsigned long>(offset));
        fflush(stderr);
        char address[32];
        snprintf(address, sizeof(address), "0x%lx",
                 static_cast<unsigned long>(offset));
        char demangle[] = "--demangle";
        char object_flag[] = "--obj";
        char *arguments[] = {frame->symbolizer, demangle, object_flag,
                             frame->object, address, nullptr};
        posix_spawn_file_actions_t actions;
        int error = posix_spawn_file_actions_init(&actions);
        if (error) {
          fprintf(stderr, "Numba ASan: cannot initialize JIT symbolizer: %d\n", error);
          break;
        }
        error = posix_spawn_file_actions_adddup2(
            &actions, STDERR_FILENO, STDOUT_FILENO);
        pid_t child;
        if (!error)
          error = posix_spawn(&child, frame->symbolizer, &actions, nullptr,
                              arguments, environ);
        posix_spawn_file_actions_destroy(&actions);
        if (!error) {
          int status = 0;
          pid_t waited;
          do {
            waited = waitpid(child, &status, 0);
          } while (waited < 0 && errno == EINTR);
          if (waited < 0 || !WIFEXITED(status) || WEXITSTATUS(status))
            fprintf(stderr, "Numba ASan: JIT symbolizer failed\n");
        } else {
          fprintf(stderr, "Numba ASan: cannot launch JIT symbolizer: %d\n", error);
        }
        break;
      }
    }
    ++line;
  }
}

extern "C" void numba_asan_install_report_callback() {
  __asan_set_error_report_callback(report_jit_frames);
}
