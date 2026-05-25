// DPMoE / core — ModelExecutor registry.

#include "executor.h"

#include <mutex>
#include <string>
#include <unordered_map>

namespace dp_moe_ext {

namespace {

std::mutex & registry_mu() {
    static std::mutex m;
    return m;
}

std::unordered_map<std::string, ExecutorFactory> & registry() {
    static std::unordered_map<std::string, ExecutorFactory> r;
    return r;
}

}  // anon

void register_executor(const char * name, ExecutorFactory factory) {
    if (name == nullptr || name[0] == '\0' || factory == nullptr) return;
    std::lock_guard<std::mutex> lk(registry_mu());
    registry()[std::string(name)] = factory;
}

std::unique_ptr<ModelExecutor> make_executor(const char * name) {
    if (name == nullptr || name[0] == '\0') return nullptr;
    std::lock_guard<std::mutex> lk(registry_mu());
    auto it = registry().find(name);
    if (it == registry().end()) return nullptr;
    return it->second();
}

}  // namespace dp_moe_ext
