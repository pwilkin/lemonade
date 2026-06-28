#include "lemon/backends/thinksound/thinksound_server.h"
#include "lemon/backends/thinksound/thinksound.h"
#include "lemon/backends/backend_registry.h"
#include "lemon/backends/backend_ops.h"
#include "lemon/model_manager.h"
#include <string>
#include <vector>

namespace lemon {
namespace backends {

InstallParams ThinkSoundServer::get_install_params(const std::string& backend, const std::string& version) {
    (void)version;
    InstallParams params;
    params.repo = "pwilkin/thinksound.cpp";
    params.filename = "thinksound-" + backend + "-x86_64.tar.gz";
    return params;
}

ThinkSoundServer::ThinkSoundServer(const std::string& log_level,
                                   ModelManager* model_manager,
                                   BackendManager* backend_manager)
    : GgmlMediaServer("thinksound-server", log_level, model_manager, backend_manager) {}

ThinkSoundServer::~ThinkSoundServer() {
    unload();
}

const BackendSpec* ThinkSoundServer::media_spec() const {
    return thinksound::spec();
}

std::vector<std::string> ThinkSoundServer::build_server_args(const ModelInfo& model_info) {
    (void)model_info;
    // The checkpoint is the directory of ThinkSound GGUFs; ts-server resolves the
    // individual networks (dit/t5/clip/vae/tokenizers) within it.
    return {
        "--dir", resolved_model_path_,
        "--host", "127.0.0.1",
        "--port", std::to_string(port_),
    };
}

void ThinkSoundServer::audio_generations(const json& request, httplib::DataSink& sink) {
    // Map the Lemonade /audio/generations request onto ts-server's /generate.
    json body;
    body["caption"] = request.value("prompt", std::string());
    body["description"] = request.contains("description") ? request["description"]
                        : request.contains("cot") ? request["cot"]
                        : body["caption"];
    if (request.contains("duration")) body["duration"] = request["duration"];
    if (request.contains("steps"))    body["steps"]    = request["steps"];
    if (request.contains("cfg"))      body["cfg"]      = request["cfg"];
    if (request.contains("seed"))     body["seed"]     = request["seed"];
    generate_to_sink("/generate", body.dump(), "application/json", sink, 600);
}

}  // namespace backends

namespace backends {
namespace thinksound {

std::unique_ptr<WrappedServer> create(const BackendContext& ctx) {
    return make_server<ThinkSoundServer>(ctx);
}

const BackendSpec* spec() { return make_spec<ThinkSoundServer>(descriptor); }
const BackendOps* ops() { return default_backend_ops(); }

}  // namespace thinksound
}  // namespace backends
}  // namespace lemon
