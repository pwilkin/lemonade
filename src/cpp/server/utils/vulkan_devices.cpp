#include "lemon/utils/vulkan_devices.h"

#ifdef HAVE_VULKAN

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace lemon {
namespace utils {

namespace {

// Load the Vulkan loader at runtime so lemond never hard-depends on libvulkan
// (CPU-only / CUDA-only hosts still link and run; this just returns "" there).
void* vk_open() {
#ifdef _WIN32
    return reinterpret_cast<void*>(LoadLibraryA("vulkan-1.dll"));
#elif defined(__APPLE__)
    void* h = dlopen("libvulkan.1.dylib", RTLD_NOW | RTLD_LOCAL);
    if (!h) h = dlopen("libvulkan.dylib", RTLD_NOW | RTLD_LOCAL);
    if (!h) h = dlopen("libMoltenVK.dylib", RTLD_NOW | RTLD_LOCAL);
    return h;
#else
    void* h = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!h) h = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    return h;
#endif
}

void* vk_sym(void* lib, const char* name) {
#ifdef _WIN32
    return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(lib), name));
#else
    return dlsym(lib, name);
#endif
}

void vk_close(void* lib) {
#ifdef _WIN32
    FreeLibrary(reinterpret_cast<HMODULE>(lib));
#else
    dlclose(lib);
#endif
}

}  // namespace

std::string vram_sorted_vulkan_device_order() {
    void* lib = vk_open();
    if (!lib) return "";

    auto fpCreate  = reinterpret_cast<PFN_vkCreateInstance>(vk_sym(lib, "vkCreateInstance"));
    auto fpDestroy = reinterpret_cast<PFN_vkDestroyInstance>(vk_sym(lib, "vkDestroyInstance"));
    auto fpEnum    = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(vk_sym(lib, "vkEnumeratePhysicalDevices"));
    auto fpProps   = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(vk_sym(lib, "vkGetPhysicalDeviceProperties"));
    auto fpMem     = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(vk_sym(lib, "vkGetPhysicalDeviceMemoryProperties"));
    if (!fpCreate || !fpDestroy || !fpEnum || !fpProps || !fpMem) { vk_close(lib); return ""; }

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "lemonade";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;

    VkInstance inst = VK_NULL_HANDLE;
    if (fpCreate(&ici, nullptr, &inst) != VK_SUCCESS) { vk_close(lib); return ""; }

    uint32_t n = 0;
    fpEnum(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    if (n) fpEnum(inst, &n, devs.data());

    struct Dev { uint32_t index; int rank; uint64_t heap; };
    std::vector<Dev> gpus;
    for (uint32_t i = 0; i < n; ++i) {
        VkPhysicalDeviceProperties pr;
        fpProps(devs[i], &pr);
        if (pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) continue;  // skip llvmpipe / software
        int rank = pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU   ? 2
                 : pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 1 : 0;
        VkPhysicalDeviceMemoryProperties mp;
        fpMem(devs[i], &mp);
        uint64_t heap = 0;
        for (uint32_t h = 0; h < mp.memoryHeapCount; ++h)
            if (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                heap = heap > mp.memoryHeaps[h].size ? heap : mp.memoryHeaps[h].size;
        gpus.push_back({i, rank, heap});
    }

    fpDestroy(inst, nullptr);
    vk_close(lib);

    if (gpus.size() < 2) return "";  // 0/1 usable GPU: nothing to reorder
    std::stable_sort(gpus.begin(), gpus.end(), [](const Dev& a, const Dev& b) {
        if (a.rank != b.rank) return a.rank > b.rank;  // discrete before integrated
        return a.heap > b.heap;                        // then roomiest first
    });
    std::string order;
    for (const auto& d : gpus) {
        if (!order.empty()) order += ",";
        order += std::to_string(d.index);
    }
    return order;
}

}  // namespace utils
}  // namespace lemon

#else  // !HAVE_VULKAN

namespace lemon {
namespace utils {
std::string vram_sorted_vulkan_device_order() { return ""; }
}  // namespace utils
}  // namespace lemon

#endif
