#include "lemon/backends/openmoss/openmoss_server.h"
#include "lemon/backends/openmoss/openmoss.h"
#include "lemon/backends/backend_registry.h"
#include "lemon/backends/backend_ops.h"
#include "lemon/model_manager.h"
#include <optional>
#include <string>
#include <vector>

namespace lemon {
namespace backends {

InstallParams OpenMossServer::get_install_params(const std::string& backend, const std::string& version) {
    (void)version;
    InstallParams params;
    params.repo = "pwilkin/openmoss";
    params.filename = media_release_asset("moss-tts", backend);
    return params;
}

OpenMossServer::OpenMossServer(const std::string& log_level,
                               ModelManager* model_manager,
                               BackendManager* backend_manager)
    : GgmlMediaServer("openmoss-server", log_level, model_manager, backend_manager) {}

OpenMossServer::~OpenMossServer() {
    unload();
}

const BackendSpec* OpenMossServer::media_spec() const {
    return openmoss::spec();
}

std::vector<std::string> OpenMossServer::build_server_args(const ModelInfo& model_info) {
    (void)model_info;
    return {
        "--model", resolved_model_path_,
        "--host", "127.0.0.1",
        "--port", std::to_string(port_),
        "--no-webui",
    };
}

void OpenMossServer::audio_speech(const json& request, httplib::DataSink& sink) {
    // moss-tts-server implements the OpenAI /v1/audio/speech schema and returns
    // audio/wav; forward the request unchanged and stream the bytes back.
    generate_to_sink("/v1/audio/speech", request.dump(), "application/json", sink, 600);
}

}  // namespace backends

namespace backends {

namespace {
// The MOSS backbone GGUF ships with a "<stem>.extras.gguf" codec sidecar that
// moss-tts-server auto-locates alongside it; fetch both.
class OpenMossOps : public BackendOps {
public:
    std::optional<std::vector<std::string>> select_checkpoint_files(
        const std::string& main_variant, const std::vector<std::string>& repo_files) const override {
        std::vector<std::string> want = {main_variant};
        auto pos = main_variant.rfind(".gguf");
        if (pos != std::string::npos) {
            std::string extras = main_variant.substr(0, pos) + ".extras.gguf";
            for (const auto& f : repo_files) {
                if (f == extras) { want.push_back(extras); break; }
            }
        }
        return want;
    }
};
}  // namespace

namespace openmoss {

std::unique_ptr<WrappedServer> create(const BackendContext& ctx) {
    return make_server<OpenMossServer>(ctx);
}

const BackendSpec* spec() { return make_spec<OpenMossServer>(descriptor); }
const BackendOps* ops() { return single_ops<OpenMossOps>(); }

}  // namespace openmoss
}  // namespace backends
}  // namespace lemon
