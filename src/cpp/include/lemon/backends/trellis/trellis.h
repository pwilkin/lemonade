#pragma once

#include "lemon/backends/backend_descriptor.h"

namespace lemon {
namespace backends {
namespace trellis {

// TRELLIS.2 image->3D generation, wrapped via its resident trellis-server
// (a GGML media engine). Serves the /3d/generations capability.
inline const BackendDescriptor descriptor = {
    /*recipe*/          "trellis",
    /*display_name*/    "TRELLIS.2",
    /*binary*/          "trellis-server",
    /*config_section*/  "",  // defaults to recipe
    /*default_device*/  DEVICE_GPU,
    /*slot_policy*/     SlotPolicy::Standard,
    /*selectable_backend*/ false,
    /*uses_ctx_size*/   false,
    /*dynamic_models*/  false,
    /*options*/ {},
    /*support*/ {
        {"vulkan", {"linux", "windows"}, {{"cpu", {"x86_64"}}, {"amd_gpu", {}}, {"nvidia_gpu", {}}}, "Vulkan-capable GPUs"},
    },
    /*default_labels*/  {"3d"},
    /*required_checkpoints*/ {"main"},
    /*modality*/        "3D generation",
    /*experimental*/    true,
    /*web_display_name*/ "",
    /*rocm_channels*/   {},
    /*exposes_prometheus_metrics*/ false,
    /*rocm_requires_cwsr_fix*/ false,
    /*version_policy*/  VersionPolicy::Exact,
    /*self_manages_downloads*/ false,
    /*takes_args*/      false,
    /*arg_variants*/    {},
    /*bin_variants*/    {"vulkan"},
    /*config_extra*/    nlohmann::json::object(),
};

}  // namespace trellis
}  // namespace backends
}  // namespace lemon
