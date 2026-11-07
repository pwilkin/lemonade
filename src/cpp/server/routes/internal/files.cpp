#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include <lemon/utils/aixlog.hpp>

#include "lemon/runtime_config.h"
#include "lemon/server/api_route.h"
#include "lemon/utils/path_utils.h"

namespace lemon {
namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;
using ordered_json = nlohmann::ordered_json;

constexpr size_t kMaxEntries = 5000;

const RouteError kNotFound{404, "", "`path` does not exist or is not a directory"};
const RouteError kNotReadable{403, "", "`path` exists but lemond is not allowed to list it"};

const RouteArg kPath{"path", ArgIn::Query, {{"type", "string"}}, false, Support::Available,
                     "Absolute directory to list. Defaults to `extra_models_dir` when it is set "
                     "and exists, otherwise to the home directory of the user running lemond."};
const RouteArg kExtensions{"extensions", ArgIn::Query, {{"type", "string"}}, false, Support::Available,
                           "Comma-separated file extensions to keep, e.g. `.gguf,.safetensors`, "
                           "compared case-insensitively. Directories are always listed. When "
                           "omitted, every file is listed."};
const RouteArg kShowHidden{"show_hidden", ArgIn::Query, {{"type", "boolean"}}, false, Support::Available,
                           "List entries whose name starts with `.`. Defaults to `false`."};

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

std::vector<std::string> parse_extensions(const std::string& raw) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= raw.size()) {
        size_t end = raw.find(',', start);
        if (end == std::string::npos) end = raw.size();
        std::string ext = raw.substr(start, end - start);
        ext.erase(0, ext.find_first_not_of(" \t"));
        ext.erase(ext.find_last_not_of(" \t") + 1);
        if (!ext.empty()) {
            if (ext.front() != '.') ext.insert(ext.begin(), '.');
            out.push_back(lower(ext));
        }
        start = end + 1;
    }
    return out;
}

bool is_existing_dir(const fs::path& p) {
    std::error_code ec;
    return fs::is_directory(p, ec) && !ec;
}

fs::path default_directory(const RuntimeConfig* config) {
    if (config) {
        const std::string extra = config->extra_models_dir();
        if (!extra.empty()) {
            fs::path p = utils::path_from_utf8(extra);
            if (is_existing_dir(p)) return p;
        }
    }
#ifdef _WIN32
    const std::string home = utils::get_environment_variable_utf8("USERPROFILE");
#else
    const std::string home = utils::get_environment_variable_utf8("HOME");
#endif
    if (!home.empty() && is_existing_dir(utils::path_from_utf8(home))) {
        return utils::path_from_utf8(home);
    }
    return fs::current_path().root_path();
}

json filesystem_roots() {
    json roots = json::array();
#ifdef _WIN32
    for (char letter = 'A'; letter <= 'Z'; ++letter) {
        const std::string drive = std::string(1, letter) + ":\\";
        if (is_existing_dir(fs::path(drive))) roots.push_back(drive);
    }
#else
    roots.push_back("/");
#endif
    return roots;
}

class FilesRoute : public ApiRoute {
public:
    using ApiRoute::ApiRoute;

    RouteSpec spec() const override {
        RouteSpec s;
        s.id = "internal.files";
        s.methods = {"GET"};
        s.paths = {"files"};
        s.prefixes = Prefixes::Internal;
        s.summary = "List a directory on the server's filesystem";
        s.description =
            "Lists the subdirectories and files of one directory on the machine `lemond` runs "
            "on, so a client can pick a local model file by a path the server can open, even "
            "when the client runs on another machine. The desktop app's custom model form uses "
            "it for its local path picker.";
        s.notes = {
            "Entries are sorted directories first, then by name. Symbolic links are followed, so "
            "a link to a directory is listed as a directory and a link to a file reports the "
            "target's size. Entries `lemond` cannot stat are left out. At most 5000 entries are "
            "returned; `truncated` says whether more were left out.",
        };
        s.args = {kPath, kExtensions, kShowHidden};

        RouteResponse response;
        response.format = ResponseFormat::Json;
        response.schema = ordered_json::parse(R"json({
            "type": "object",
            "required": ["path", "parent", "separator", "roots", "entries", "truncated"],
            "properties": {
                "path": {"type": "string", "description": "The listed directory, absolute and normalized."},
                "parent": {"type": ["string", "null"], "description": "The parent directory, or `null` at a filesystem root."},
                "separator": {"type": "string", "description": "The server's path separator: `/`, or `\\` on Windows."},
                "roots": {"type": "array", "items": {"type": "string"}, "description": "Filesystem roots to start from: `/`, or the drive letters on Windows."},
                "entries": {"type": "array", "description": "The directory's entries.", "items": {
                    "type": "object",
                    "required": ["name", "path", "kind", "size_bytes"],
                    "properties": {
                        "name": {"type": "string", "description": "Entry name."},
                        "path": {"type": "string", "description": "Absolute path of the entry."},
                        "kind": {"enum": ["directory", "file"], "description": "Whether the entry is a directory or a file."},
                        "size_bytes": {"type": "integer", "description": "File size; `0` for a directory."}
                    }
                }},
                "truncated": {"type": "boolean", "description": "Whether entries past the 5000-entry limit were left out."}
            }
        })json");
        s.responses = {response};

        RouteExample example;
        example.format = ResponseFormat::Json;
        example.args = {{"path", "/usr"}, {"extensions", ".gguf,.safetensors"}};
        s.examples = {example};
        s.errors = {kNotFound, kNotReadable, kUnexpectedException};
        s.quiet_log = true;
        return s;
    }

    void handle(RouteRequest& req, httplib::Response& res) override {
        try {
            const std::string raw_path = req.value(kPath, "");
            fs::path dir = raw_path.empty() ? default_directory(ctx_.config)
                                            : utils::path_from_utf8(raw_path);
            std::error_code ec;
            if (dir.is_relative()) dir = fs::absolute(dir, ec);
            dir = dir.lexically_normal();
            if (!dir.has_filename() && dir != dir.root_path()) dir = dir.parent_path();

            std::error_code status_ec;
            const fs::file_status status = fs::status(dir, status_ec);
            if (status_ec == std::errc::permission_denied) {
                write_plain_error(res, kNotReadable, "Permission denied: " + utils::path_to_utf8(dir));
                return;
            }
            if (status_ec || !fs::is_directory(status)) {
                write_plain_error(res, kNotFound, "No such directory: " + utils::path_to_utf8(dir));
                return;
            }

            const std::vector<std::string> extensions = parse_extensions(req.value(kExtensions, ""));
            const bool show_hidden = req.value<bool>(kShowHidden, false);

            json directories = json::array();
            json files = json::array();
            bool truncated = false;

            std::error_code iter_ec;
            fs::directory_iterator it(dir, iter_ec);
            if (iter_ec) {
                write_plain_error(res, iter_ec == std::errc::permission_denied ? kNotReadable : kNotFound,
                                  "Cannot list " + utils::path_to_utf8(dir) + ": " + iter_ec.message());
                return;
            }
            for (const fs::directory_iterator end; it != end; it.increment(iter_ec)) {
                if (iter_ec) break;
                const fs::path& entry_path = it->path();
                const std::string name = utils::path_to_utf8(entry_path.filename());
                if (!show_hidden && !name.empty() && name.front() == '.') continue;

                std::error_code entry_ec;
                const fs::file_status entry_status = fs::status(entry_path, entry_ec);
                if (entry_ec) continue;

                if (fs::is_directory(entry_status)) {
                    directories.push_back({{"name", name},
                                           {"path", utils::path_to_utf8(entry_path)},
                                           {"kind", "directory"},
                                           {"size_bytes", 0}});
                } else if (fs::is_regular_file(entry_status)) {
                    if (!extensions.empty()) {
                        const std::string ext = lower(utils::path_to_utf8(entry_path.extension()));
                        if (std::find(extensions.begin(), extensions.end(), ext) == extensions.end()) continue;
                    }
                    std::error_code size_ec;
                    const auto size = fs::file_size(entry_path, size_ec);
                    files.push_back({{"name", name},
                                     {"path", utils::path_to_utf8(entry_path)},
                                     {"kind", "file"},
                                     {"size_bytes", size_ec ? 0 : static_cast<uint64_t>(size)}});
                }
                if (directories.size() + files.size() >= kMaxEntries) {
                    truncated = true;
                    break;
                }
            }

            auto by_name = [](const json& a, const json& b) {
                return lower(a["name"].get<std::string>()) < lower(b["name"].get<std::string>());
            };
            std::sort(directories.begin(), directories.end(), by_name);
            std::sort(files.begin(), files.end(), by_name);
            json entries = std::move(directories);
            for (auto& f : files) entries.push_back(std::move(f));

            const fs::path parent = dir.parent_path();
            json body = {
                {"path", utils::path_to_utf8(dir)},
                {"parent", (dir == dir.root_path() || parent == dir) ? json(nullptr) : json(utils::path_to_utf8(parent))},
#ifdef _WIN32
                {"separator", "\\"},
#else
                {"separator", "/"},
#endif
                {"roots", filesystem_roots()},
                {"entries", std::move(entries)},
                {"truncated", truncated},
            };
            res.set_content(body.dump(), "application/json");
        } catch (const std::exception& e) {
            LOG(ERROR, "Server") << "ERROR in /internal/files: " << e.what() << std::endl;
            write_plain_error(res, kUnexpectedException, e.what());
        }
    }
};

} // namespace

std::unique_ptr<ApiRoute> make_files_route(ServerContext& ctx) {
    return std::make_unique<FilesRoute>(ctx);
}

} // namespace lemon
