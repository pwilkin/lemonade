#pragma once

#include "lemon/backends/backend_registry.h"

#include "lemon/ggml_media_server.h"
#include "lemon/server_capabilities.h"
#include "lemon/backends/backend_utils.h"
#include <string>
#include <vector>

namespace lemon {
namespace backends {

// ThinkSound SFX backend. Wraps the resident ts-server; maps the
// /audio/generations request onto its POST /generate.
class ThinkSoundServer : public GgmlMediaServer, public IAudioGenerationServer {
public:
    static InstallParams get_install_params(const std::string& backend, const std::string& version);

    ThinkSoundServer(const std::string& log_level,
                     ModelManager* model_manager,
                     BackendManager* backend_manager);
    ~ThinkSoundServer() override;

    // IAudioGenerationServer
    void audio_generations(const json& request, httplib::DataSink& sink) override;

protected:
    const BackendSpec* media_spec() const override;
    std::vector<std::string> build_server_args(const ModelInfo& model_info) override;
};

namespace thinksound {
std::unique_ptr<WrappedServer> create(const BackendContext& ctx);
const BackendSpec* spec();
const BackendOps* ops();
}  // namespace thinksound

}  // namespace backends
}  // namespace lemon
