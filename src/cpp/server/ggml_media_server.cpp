#include "lemon/ggml_media_server.h"

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>
#include "lemon/backend_manager.h"
#include "lemon/backends/backend_utils.h"
#include "lemon/backends/hf_cache_util.h"
#include "lemon/model_manager.h"
#include "lemon/utils/path_utils.h"
#include "lemon/utils/process_manager.h"
#include "lemon/utils/vulkan_devices.h"
#include "lemon/runtime_config.h"
#include "lemon/system_info.h"
#include <lemon/utils/aixlog.hpp>

namespace lemon {

using backends::BackendUtils;

std::string GgmlMediaServer::media_backend_variant() const {
    // Opt-in backend selection via "<section>.backend" (default "vulkan", so the
    // working Vulkan path is unchanged). "rocm" pulls the ROCm build and uses the
    // shared TheRock runtime (like sd-cpp / llama.cpp).
    if (auto* cfg = RuntimeConfig::global()) {
        const std::string section =
            RuntimeConfig::recipe_to_config_section(media_spec()->recipe);
        const std::string b = cfg->backend_string(section, "backend");
        if (!b.empty() && b != "auto") {
            return b;
        }
    }
    return "vulkan";
}

std::vector<std::pair<std::string, std::string>> GgmlMediaServer::build_server_env() {
    // Route ggml-vulkan media backends (ace-server, ts-server, ...) to the GPU with
    // the most VRAM. ggml-vulkan otherwise uses Vulkan device 0, which on a mixed
    // multi-GPU box can be the smaller card and OOM on a large generation (long
    // audio, big meshes). vram_sorted_vulkan_device_order() enumerates the Vulkan
    // devices (any vendor), drops software rasterizers, and orders them so the
    // roomiest real GPU is logical device 0. Empty -> keep ggml's default selection.
    std::vector<std::pair<std::string, std::string>> env;

    // ROCm: the slim binary finds the ROCm runtime in the shared TheRock SDK
    // (downloaded per the user's GPU arch) plus its own colocated ggml .so. Mirror
    // sd-cpp/llama.cpp: prepend the TheRock lib dir + the exe dir to the loader path.
    if (media_backend_variant() == "rocm") {
        const std::string arch = SystemInfo::get_rocm_arch();
        const std::string therock_lib = arch.empty() ? "" : BackendUtils::get_therock_lib_path(arch);
        const std::string exe_dir = std::filesystem::path(exe_path_).parent_path().string();
#ifdef _WIN32
        std::string path = therock_lib.empty() ? exe_dir : (therock_lib + ";" + exe_dir);
        if (const char* p = std::getenv("PATH")) path += std::string(";") + p;
        env.push_back({"PATH", path});
#else
        std::string ld = therock_lib.empty() ? exe_dir : (therock_lib + ":" + exe_dir);
        if (const char* p = std::getenv("LD_LIBRARY_PATH")) ld += std::string(":") + p;
        env.push_back({"LD_LIBRARY_PATH", ld});
#endif
        return env;
    }

    // Route ggml-vulkan media backends (ace-server, ts-server, ...) to the GPU with
    // the most VRAM. ggml-vulkan otherwise uses Vulkan device 0, which on a mixed
    // multi-GPU box can be the smaller card and OOM on a large generation (long
    // audio, big meshes). vram_sorted_vulkan_device_order() enumerates the Vulkan
    // devices (any vendor), drops software rasterizers, and orders them so the
    // roomiest real GPU is logical device 0. Empty -> keep ggml's default selection.
    std::string order = utils::vram_sorted_vulkan_device_order();
    if (!order.empty()) {
        LOG(INFO, server_name_) << "routing to GPUs by VRAM (most first): GGML_VK_VISIBLE_DEVICES="
                                << order << std::endl;
        env.push_back({"GGML_VK_VISIBLE_DEVICES", order});
    }
    return env;
}

GgmlMediaServer::GgmlMediaServer(const std::string& server_name,
                                 const std::string& log_level,
                                 ModelManager* model_manager,
                                 BackendManager* backend_manager)
    : WrappedServer(server_name, log_level, model_manager, backend_manager) {}

GgmlMediaServer::~GgmlMediaServer() {
    GgmlMediaServer::unload();
}

std::string GgmlMediaServer::resolve_binary_path() {
    const backends::BackendSpec* spec = media_spec();
    const std::string backend = media_backend_variant();

    // An explicit "<variant>_bin" path in config wins so a locally-built binary
    // can be used without a published release. Only then do we install.
    std::string external = BackendUtils::find_external_backend_binary(spec->recipe, backend);
    if (!external.empty() && std::filesystem::exists(external)) {
        return external;
    }

    backend_manager_->install_backend(spec->recipe, backend);
    return BackendUtils::get_backend_binary_path(*spec, backend);
}

void GgmlMediaServer::load(const std::string& model_name,
                           const ModelInfo& model_info,
                           const RecipeOptions& options,
                           bool do_not_upgrade) {
    (void)options;
    (void)do_not_upgrade;
    LOG(INFO, server_name_) << "Loading model: " << model_name << std::endl;

    resolved_model_path_ = model_info.resolved_path();
    if (resolved_model_path_.empty() || !std::filesystem::exists(resolved_model_path_)) {
        throw std::runtime_error("Model path not found for checkpoint: " + model_info.checkpoint());
    }

    exe_path_ = resolve_binary_path();

    port_ = choose_port();
    if (port_ == 0) {
        throw std::runtime_error("Failed to find an available port");
    }

    std::vector<std::string> args = build_server_args(model_info);
    std::vector<std::pair<std::string, std::string>> env_vars = build_server_env();

    LOG(INFO, server_name_) << "Starting " << exe_path_ << " on port " << port_ << std::endl;

    ProcessHandle started_handle = utils::ProcessManager::start_process(
        exe_path_, args, "", is_debug(), false, env_vars);
    set_process_handle(started_handle);

    if (!has_process_handle(started_handle)) {
        throw std::runtime_error("Failed to start " + server_name_ + " process");
    }
    LOG(INFO, server_name_) << "Process started with PID: " << started_handle.pid << std::endl;

    if (!wait_for_ready(health_endpoint())) {
        unload();
        throw std::runtime_error(server_name_ + " failed to start or become ready");
    }
}

void GgmlMediaServer::unload() {
    stop_backend_watchdog();
    const ProcessHandle handle = consume_process_handle_for_cleanup();
    if (has_process_handle(handle)) {
        LOG(INFO, server_name_) << "Stopping server (PID: " << handle.pid << ")" << std::endl;
        utils::ProcessManager::stop_process(handle);
    }
}

// Stream the response bytes to the sink, or — on a backend error or transport
// failure — log it and write nothing. Crucially this never throws: it runs
// inside the router's streaming content provider, where an escaping exception
// would terminate the whole process.
void GgmlMediaServer::deliver(const utils::HttpResponse& resp, httplib::DataSink& sink) {
    if (resp.curl_code != 0) {
        LOG(ERROR, server_name_) << "generation transport error: " << resp.curl_error << std::endl;
        return;
    }
    if (resp.status_code < 200 || resp.status_code >= 300) {
        LOG(ERROR, server_name_) << "generation failed (HTTP " << resp.status_code
                                 << "): " << resp.body << std::endl;
        return;
    }
    sink.write(resp.body.data(), resp.body.size());
}

void GgmlMediaServer::generate_to_sink(const std::string& endpoint,
                                       const std::string& body,
                                       const std::string& content_type,
                                       httplib::DataSink& sink,
                                       long timeout_seconds) {
    // Generation is a single slow call (seconds to minutes); allow plenty of time.
    const long timeout = timeout_seconds > 0 ? timeout_seconds : 1800;
    try {
        deliver(utils::HttpClient::post(get_base_url() + endpoint, body,
                                        {{"Content-Type", content_type}}, timeout),
                sink);
    } catch (const std::exception& e) {
        LOG(ERROR, server_name_) << "generation request failed: " << e.what() << std::endl;
    }
}

void GgmlMediaServer::generate_multipart_to_sink(const std::string& endpoint,
                                                 const std::vector<utils::MultipartField>& fields,
                                                 httplib::DataSink& sink,
                                                 long timeout_seconds) {
    const long timeout = timeout_seconds > 0 ? timeout_seconds : 1800;
    try {
        deliver(utils::HttpClient::post_multipart(get_base_url() + endpoint, fields, timeout), sink);
    } catch (const std::exception& e) {
        LOG(ERROR, server_name_) << "generation request failed: " << e.what() << std::endl;
    }
}

namespace backends {

std::string GgmlMediaDirOps::resolve_checkpoint_path(const ModelInfo& info,
                                                     const CheckpointResolveContext& ctx) const {
    (void)info;
    // The binary wants the folder that holds the GGUFs — i.e. the active HF
    // snapshot, not the repo cache root the base class returns for no-variant
    // checkpoints.
    std::filesystem::path root = lemon::utils::path_from_utf8(ctx.model_cache_path);
    std::filesystem::path snap = hf_cache::active_snapshot_path(root);
    if (!snap.empty() && hf_cache::exists(snap)) {
        return lemon::utils::path_to_utf8(snap);
    }
    return ctx.model_cache_path;
}

} // namespace backends

} // namespace lemon
