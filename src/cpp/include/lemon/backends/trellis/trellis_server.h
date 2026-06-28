#pragma once

#include "lemon/backends/backend_registry.h"

#include "lemon/ggml_media_server.h"
#include "lemon/server_capabilities.h"
#include "lemon/backends/backend_utils.h"
#include <string>
#include <vector>

namespace lemon {
namespace backends {

// TRELLIS.2 image->3D backend. Wraps the resident trellis-server; forwards the
// /3d/generations input image as multipart and streams back the GLB.
class TrellisServer : public GgmlMediaServer, public IModel3DServer {
public:
    static InstallParams get_install_params(const std::string& backend, const std::string& version);

    TrellisServer(const std::string& log_level,
                  ModelManager* model_manager,
                  BackendManager* backend_manager);
    ~TrellisServer() override;

    // IModel3DServer
    void model_3d_generations(const json& request, httplib::DataSink& sink) override;

protected:
    const BackendSpec* media_spec() const override;
    std::vector<std::string> build_server_args(const ModelInfo& model_info) override;
    std::vector<std::pair<std::string, std::string>> build_server_env() override;
};

namespace trellis {
std::unique_ptr<WrappedServer> create(const BackendContext& ctx);
const BackendSpec* spec();
const BackendOps* ops();
}  // namespace trellis

}  // namespace backends
}  // namespace lemon
