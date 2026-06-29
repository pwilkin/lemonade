#pragma once

#include <string>
#include <utility>
#include <vector>
#include <httplib.h>
#include "lemon/backends/backend_ops.h"
#include "lemon/backends/backend_utils.h"
#include "lemon/utils/http_client.h"
#include "lemon/wrapped_server.h"

namespace lemon {

// Generic base for backends that wrap a *resident* GGML HTTP-server subprocess —
// a "GGML media engine". One binary loads its model(s) once and answers
// generation requests over HTTP for as long as the model stays hot; the router's
// existing LRU/eviction machinery manages its lifetime like any other model.
//
// Concrete backends subclass this, additionally inherit the capability
// interface(s) they serve (IAudioGenerationServer, IModel3DServer,
// ITextToSpeechServer, …), and fill in a few small policy hooks: how to launch
// the subprocess and how to map a Lemonade request onto the subprocess's API.
// The capability method bodies are tiny — they call generate_to_sink() /
// generate_multipart_to_sink() to forward the request and stream the raw media
// bytes back to the client.
class GgmlMediaServer : public WrappedServer {
public:
    GgmlMediaServer(const std::string& server_name,
                    const std::string& log_level,
                    ModelManager* model_manager,
                    BackendManager* backend_manager);
    ~GgmlMediaServer() override;

    void load(const std::string& model_name,
              const ModelInfo& model_info,
              const RecipeOptions& options,
              bool do_not_upgrade = false) override;

    void unload() override;

protected:
    // ---- policy hooks (each backend overrides) ----

    // The backend recipe's install/download spec (binary name + install params).
    virtual const backends::BackendSpec* media_spec() const = 0;

    // The backend variant to resolve/install (the config key is
    // "<section>.<variant>_bin"). Defaults to "vulkan" — Lemonade is AMD-centric
    // and these tools run their GGML graphs on a Vulkan device.
    virtual std::string media_backend_variant() const { return "vulkan"; }

    // CLI args for the resident server subprocess. Built from the resolved model
    // path (resolved_model_path_) and the chosen port (port_). The binary itself
    // is prepended by ProcessManager, so do NOT include it here.
    virtual std::vector<std::string> build_server_args(const ModelInfo& model_info) = 0;

    // Extra environment for the subprocess. The base implementation routes
    // ggml-vulkan backends to the GPU with the most VRAM (GGML_VK_VISIBLE_DEVICES),
    // so a long generation doesn't land on a smaller card and OOM. Overrides that
    // want this should call GgmlMediaServer::build_server_env() and append.
    virtual std::vector<std::pair<std::string, std::string>> build_server_env();

    // Endpoint polled by wait_for_ready() to detect server startup.
    virtual std::string health_endpoint() const { return "/health"; }

    // ---- helpers for capability methods ----

    // POST a JSON body to `endpoint` on the resident subprocess and stream the
    // raw response bytes (audio/glb/…) to `sink`. Throws on a non-2xx reply.
    void generate_to_sink(const std::string& endpoint,
                          const std::string& body,
                          const std::string& content_type,
                          httplib::DataSink& sink,
                          long timeout_seconds = 0);

    // As above, with a multipart form-data request (e.g. an input image).
    void generate_multipart_to_sink(const std::string& endpoint,
                                     const std::vector<utils::MultipartField>& fields,
                                     httplib::DataSink& sink,
                                     long timeout_seconds = 0);

    // Resolve the subprocess binary: an explicit "<variant>_bin" path in config
    // wins (lets a locally-built binary be used with no published release),
    // otherwise install the managed binary and resolve it. Overridable.
    virtual std::string resolve_binary_path();

    std::string resolved_model_path_;  // absolute path to the model file/dir
    std::string exe_path_;             // resolved subprocess binary

private:
    // Write a backend response to the sink, or log-and-drop on error. Never
    // throws — runs inside the router's streaming content provider.
    void deliver(const utils::HttpResponse& resp, httplib::DataSink& sink);
};

namespace backends {

// Shared ops for directory-style GGML model repos (a folder of GGUFs that the
// backend binary loads via --dir/--models). Resolves the checkpoint to the
// active Hugging Face snapshot directory rather than the cache root, so the
// resolved path actually contains the downloaded files.
class GgmlMediaDirOps : public BackendOps {
public:
    std::string resolve_checkpoint_path(const ModelInfo& info,
                                        const CheckpointResolveContext& ctx) const override;
};

} // namespace backends

} // namespace lemon
