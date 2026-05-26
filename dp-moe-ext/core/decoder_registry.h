// DPMoE / core — decoder factory boundary.
//
// Core owns runtime mechanics and asks the decoder layer to wrap a
// parsed UpstreamLayoutHost in the right ChunkedTensor subclass.

#pragma once

#include "chunked_tensor.h"
#include "upstream_layout.h"

#include <memory>
#include <string>

namespace dp_moe_ext {

std::unique_ptr<ChunkedTensor> make_chunked_tensor(
    const std::string & wid,
    UpstreamLayoutHost host);

}  // namespace dp_moe_ext
