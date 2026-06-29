#include "lemon/backends/trellis/trellis_server.h"
#include "lemon/backends/trellis/trellis.h"
#include "lemon/backends/backend_registry.h"
#include "lemon/backends/backend_ops.h"
#include "lemon/model_manager.h"
#include "lemon/utils/json_utils.h"
#include <string>
#include <vector>

namespace lemon {
namespace backends {

InstallParams TrellisServer::get_install_params(const std::string& backend, const std::string& version) {
    (void)version;
    InstallParams params;
    params.repo = "pwilkin/trellis.cpp";
    params.filename = "trellis-" + backend + "-x86_64.tar.gz";
    return params;
}

TrellisServer::TrellisServer(const std::string& log_level,
                             ModelManager* model_manager,
                             BackendManager* backend_manager)
    : GgmlMediaServer("trellis-server", log_level, model_manager, backend_manager) {}

TrellisServer::~TrellisServer() {
    unload();
}

const BackendSpec* TrellisServer::media_spec() const {
    return trellis::spec();
}

std::vector<std::string> TrellisServer::build_server_args(const ModelInfo& model_info) {
    (void)model_info;
    // The checkpoint is the directory of TRELLIS.2 GGUFs.
    return {
        "--models", resolved_model_path_,
        "--host", "127.0.0.1",
        "--port", std::to_string(port_),
    };
}

std::vector<std::pair<std::string, std::string>> TrellisServer::build_server_env() {
    // Run the 512 cascade (TRELLIS_512=1): it skips the _1024 flow models and uses
    // smaller grids/meshes, so peak host RAM (~15 GB intermediates for the 1024
    // cascade) and VRAM drop substantially. The 1024 cascade can OOM a memory-
    // contended box; 512 is the safe default for the prototype.
    return {{"TRELLIS_512", "1"}};
}

void TrellisServer::model_3d_generations(const json& request, httplib::DataSink& sink) {
    if (!request.contains("image") || !request["image"].is_string()) {
        return;  // handler already validated; nothing to stream
    }
    std::string b64 = request["image"].get<std::string>();
    // Strip an optional data URL prefix ("data:image/png;base64,").
    auto comma = b64.find(',');
    if (b64.rfind("data:", 0) == 0 && comma != std::string::npos) {
        b64 = b64.substr(comma + 1);
    }
    std::string image = utils::JsonUtils::base64_decode(b64);

    std::vector<utils::MultipartField> fields;
    fields.push_back({"image", image, "input.png", "image/png"});
    if (request.contains("seed")) {
        fields.push_back({"seed", std::to_string(request["seed"].get<int>()), "", ""});
    }
    // Cascade resolution (512/1024/1536); trellis-server defaults to 512 if absent.
    if (request.contains("resolution")) {
        std::string res = request["resolution"].is_string()
                              ? request["resolution"].get<std::string>()
                              : std::to_string(request["resolution"].get<int>());
        fields.push_back({"resolution", res, "", ""});
    }
    // Background removal mode (threshold | birefnet); uploads default to birefnet
    // on the client so an arbitrary photo's real background gets matted out.
    if (request.contains("bg_removal") && request["bg_removal"].is_string()) {
        fields.push_back({"bg_removal", request["bg_removal"].get<std::string>(), "", ""});
    }
    // 3D reconstruction is slow (the 1024 cascade is minutes); allow ample time.
    generate_multipart_to_sink("/generate", fields, sink, 1800);
}

}  // namespace backends

namespace backends {
namespace trellis {

std::unique_ptr<WrappedServer> create(const BackendContext& ctx) {
    return make_server<TrellisServer>(ctx);
}

const BackendSpec* spec() { return make_spec<TrellisServer>(descriptor); }
const BackendOps* ops() { return single_ops<GgmlMediaDirOps>(); }

}  // namespace trellis
}  // namespace backends
}  // namespace lemon
