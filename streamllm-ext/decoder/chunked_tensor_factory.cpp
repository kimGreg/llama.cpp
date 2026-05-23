#include "core/decoder_registry.h"

#include "decoder/anybcq/tensor.h"
#include "decoder/direct_matrix/tensor.h"

#include <stdexcept>
#include <utility>

namespace streamllm_ext {

std::unique_ptr<ChunkedTensor> make_chunked_tensor(
    const std::string & wid,
    UpstreamLayoutHost host) {
    if (host.direct_matrix) {
        return std::unique_ptr<ChunkedTensor>(
            new direct_matrix::DirectMatrixTensor(wid, std::move(host)));
    }
    return wrap_host_in_tensor(wid, std::move(host));
}

}  // namespace streamllm_ext
