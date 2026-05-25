// DPMoE / decoder / ss_anybcq — concrete ChunkedTensor for
// the SS_ANYBCQ AnyBCQ encoding.
//
// Inherits behavior from anybcq::AnyBCQFamilyTensor: both encoders
// register their byte-format-specific transform / after-load / after-
// evict callbacks on the shared UpstreamLayoutHost, and the family
// tensor's virtuals delegate to those callbacks. The distinct C++ type
// is purely a family tag — call sites that explicitly want the ss_anybcq
// (single-α-per-chunk) variant can dynamic_cast / static_cast onto it.

#pragma once

#include "../anybcq/tensor.h"

namespace dp_moe_ext { namespace ss_anybcq {

class SsAnybcqTensor : public anybcq::AnyBCQFamilyTensor {
public:
    using anybcq::AnyBCQFamilyTensor::AnyBCQFamilyTensor;
};

}}  // namespace dp_moe_ext::ss_anybcq
