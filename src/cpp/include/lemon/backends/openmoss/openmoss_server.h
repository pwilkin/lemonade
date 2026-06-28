#pragma once

#include "lemon/backends/backend_registry.h"

#include "lemon/ggml_media_server.h"
#include "lemon/server_capabilities.h"
#include "lemon/backends/backend_utils.h"
#include <string>
#include <vector>

namespace lemon {
namespace backends {

// OpenMOSS TTS backend. Wraps the resident moss-tts-server, which already speaks
// the OpenAI /v1/audio/speech schema, so audio_speech() just forwards.
class OpenMossServer : public GgmlMediaServer, public ITextToSpeechServer {
public:
    static InstallParams get_install_params(const std::string& backend, const std::string& version);

    OpenMossServer(const std::string& log_level,
                   ModelManager* model_manager,
                   BackendManager* backend_manager);
    ~OpenMossServer() override;

    // ITextToSpeechServer
    void audio_speech(const json& request, httplib::DataSink& sink) override;

protected:
    const BackendSpec* media_spec() const override;
    std::vector<std::string> build_server_args(const ModelInfo& model_info) override;
};

namespace openmoss {
std::unique_ptr<WrappedServer> create(const BackendContext& ctx);
const BackendSpec* spec();
const BackendOps* ops();
}  // namespace openmoss

}  // namespace backends
}  // namespace lemon
