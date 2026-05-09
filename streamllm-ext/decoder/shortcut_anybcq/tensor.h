// streamllm-ext / decoder / shortcut_anybcq — concrete ChunkedTensor for
// the SHORTCUT AnyBCQ encoding.
//
// Inherits behavior from anybcq::AnyBCQFamilyTensor: both encoders
// register their byte-format-specific transform / after-load / after-
// evict callbacks on the shared UpstreamLayoutHost, and the family
// tensor's virtuals delegate to those callbacks. The distinct C++ type
// is purely a family tag — call sites that explicitly want the shortcut
// (single-α-per-chunk) variant can dynamic_cast / static_cast onto it.

#pragma once

#include "../anybcq/tensor.h"

namespace streamllm_ext { namespace shortcut_anybcq {

class ShortcutTensor : public anybcq::AnyBCQFamilyTensor {
public:
    using anybcq::AnyBCQFamilyTensor::AnyBCQFamilyTensor;
};

}}  // namespace streamllm_ext::shortcut_anybcq
