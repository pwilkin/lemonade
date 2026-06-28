#pragma once

#include "lemon/backends/backend_registry.h"

#include "lemon/ggml_media_server.h"
#include "lemon/server_capabilities.h"
#include "lemon/backends/backend_utils.h"
#include <string>
#include <vector>

namespace lemon {
namespace backends {

// ACE-Step music backend. Wraps the resident ace-server, whose compute API is
// asynchronous (POST /synth -> job id, poll GET /job), so audio_generations()
// drives the submit/poll cycle and extracts the audio from the multipart result.
class AceStepServer : public GgmlMediaServer, public IAudioGenerationServer {
public:
    static InstallParams get_install_params(const std::string& backend, const std::string& version);

    AceStepServer(const std::string& log_level,
                  ModelManager* model_manager,
                  BackendManager* backend_manager);
    ~AceStepServer() override;

    // IAudioGenerationServer
    void audio_generations(const json& request, httplib::DataSink& sink) override;

protected:
    const BackendSpec* media_spec() const override;
    std::vector<std::string> build_server_args(const ModelInfo& model_info) override;
};

namespace acestep {
std::unique_ptr<WrappedServer> create(const BackendContext& ctx);
const BackendSpec* spec();
const BackendOps* ops();
}  // namespace acestep

}  // namespace backends
}  // namespace lemon
