#include "lemon/backends/acestep/acestep_server.h"
#include "lemon/backends/acestep/acestep.h"
#include "lemon/backends/backend_registry.h"
#include "lemon/backends/backend_ops.h"
#include "lemon/model_manager.h"
#include "lemon/utils/http_client.h"
#include <lemon/utils/aixlog.hpp>
#include <algorithm>
#include <chrono>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace lemon {
namespace backends {

namespace {
// ace-server returns synth results as multipart/mixed (audio part + latent
// part). Extract the first audio part's raw bytes. Returns "" if not found.
std::string extract_multipart_audio(const std::string& body) {
    const std::string ct = "Content-Type: audio/";
    size_t hpos = body.find(ct);
    if (hpos == std::string::npos) return "";
    size_t bstart = body.find("\r\n\r\n", hpos);
    if (bstart == std::string::npos) return "";
    bstart += 4;
    size_t bend = body.find("\r\n--ace-batch-boundary", bstart);
    if (bend == std::string::npos) return "";
    return body.substr(bstart, bend - bstart);
}
}  // namespace

InstallParams AceStepServer::get_install_params(const std::string& backend, const std::string& version) {
    (void)version;
    InstallParams params;
    params.repo = "pwilkin/acestep.cpp";
    params.filename = media_release_asset("acestep", backend);
    return params;
}

AceStepServer::AceStepServer(const std::string& log_level,
                             ModelManager* model_manager,
                             BackendManager* backend_manager)
    : GgmlMediaServer("acestep-server", log_level, model_manager, backend_manager) {}

AceStepServer::~AceStepServer() {
    unload();
}

const BackendSpec* AceStepServer::media_spec() const {
    return acestep::spec();
}

std::vector<std::string> AceStepServer::build_server_args(const ModelInfo& model_info) {
    (void)model_info;
    // The checkpoint is the directory of ACE-Step GGUFs; ace-server scans it by
    // architecture. --keep-loaded keeps the model resident (hot) across requests.
    return {
        "--models", resolved_model_path_,
        "--host", "127.0.0.1",
        "--port", std::to_string(port_),
        "--keep-loaded",
    };
}

void AceStepServer::audio_generations(const json& request, httplib::DataSink& sink) {
    try {
        // Map the Lemonade request onto ace-server's /synth (instrumental, DiT-only).
        json synth;
        synth["caption"] = request.value("prompt", std::string());
        synth["synth_model"] = "";
        synth["output_format"] = "wav24";
        if (request.contains("duration")) synth["duration"] = request["duration"];
        if (request.contains("seed"))     synth["seed"]     = request["seed"];
        if (request.contains("steps"))    synth["inference_steps"] = request["steps"];

        const std::string base = get_base_url();
        auto submit = utils::HttpClient::post(base + "/synth", synth.dump(),
                                              {{"Content-Type", "application/json"}}, 60);
        if (submit.status_code != 200) {
            LOG(ERROR, server_name_) << "synth submit failed (HTTP " << submit.status_code
                                     << "): " << submit.body << std::endl;
            return;
        }
        std::string job_id = json::parse(submit.body).value("id", std::string());
        if (job_id.empty()) {
            LOG(ERROR, server_name_) << "synth submit returned no job id: " << submit.body << std::endl;
            return;
        }

        // Poll until the job finishes (music generation: seconds to a few minutes).
        const std::string job_url = base + "/job?id=" + job_id;
        std::string status;
        for (int i = 0; i < 1200; ++i) {  // ~20 min ceiling at 1s cadence
            auto poll = utils::HttpClient::get(job_url, {}, 30);
            if (poll.status_code == 200) {
                try { status = json::parse(poll.body).value("status", std::string()); }
                catch (...) { status.clear(); }
            }
            if (status == "done" || status == "failed" || status == "cancelled") break;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        if (status != "done") {
            LOG(ERROR, server_name_) << "synth job " << job_id << " ended with status '" << status << "'" << std::endl;
            return;
        }

        auto result = utils::HttpClient::get(job_url + "&result=1", {}, 120);
        if (result.status_code != 200) {
            LOG(ERROR, server_name_) << "fetching job result failed (HTTP " << result.status_code << ")" << std::endl;
            return;
        }
        std::string audio = extract_multipart_audio(result.body);
        if (audio.empty()) {
            LOG(ERROR, server_name_) << "no audio part in multipart result" << std::endl;
            return;
        }
        sink.write(audio.data(), audio.size());
    } catch (const std::exception& e) {
        LOG(ERROR, server_name_) << "audio_generations failed: " << e.what() << std::endl;
    }
}

}  // namespace backends

namespace backends {

namespace {
// ACE-Step's HF repo holds many variants; the checkpoint variant names the DiT,
// and we additionally fetch one LM, the text encoder, and the VAE. Resolves to
// the snapshot directory (ace-server scans --models), via GgmlMediaDirOps.
class AceStepOps : public GgmlMediaDirOps {
public:
    std::optional<std::vector<std::string>> select_checkpoint_files(
        const std::string& main_variant, const std::vector<std::string>& repo_files) const override {
        static const std::vector<std::string> kCompanions = {
            "acestep-5Hz-lm-4B-Q8_0.gguf",     // language model (vocals/auto-lyrics)
            "Qwen3-Embedding-0.6B-BF16.gguf",  // text encoder
            "vae-BF16.gguf",                   // VAE
        };
        std::vector<std::string> want = {main_variant};
        for (const auto& c : kCompanions) {
            if (std::find(repo_files.begin(), repo_files.end(), c) != repo_files.end()) {
                want.push_back(c);
            }
        }
        return want;
    }
};
}  // namespace

namespace acestep {

std::unique_ptr<WrappedServer> create(const BackendContext& ctx) {
    return make_server<AceStepServer>(ctx);
}

const BackendSpec* spec() { return make_spec<AceStepServer>(descriptor); }
const BackendOps* ops() { return single_ops<AceStepOps>(); }

}  // namespace acestep
}  // namespace backends
}  // namespace lemon
