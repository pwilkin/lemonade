#pragma once

#include <string>

namespace lemon {
namespace utils {

// Returns a comma-separated list of Vulkan physical-device indices ordered for
// GGML_VK_VISIBLE_DEVICES: real GPUs only (software rasterizers like llvmpipe are
// dropped), discrete GPUs before integrated, then by device-local VRAM descending.
// So logical device 0 becomes the roomiest real GPU. Empty when there is 0/1 usable
// GPU or Vulkan is unavailable at build/runtime — callers then keep ggml's default
// device selection. Vendor-agnostic (NVIDIA/AMD/Intel): it enumerates Vulkan itself
// rather than relying on vendor-specific tooling.
std::string vram_sorted_vulkan_device_order();

}  // namespace utils
}  // namespace lemon
