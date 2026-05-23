#pragma once

#include "chunked_tensor.h"

namespace streamllm_ext {

void apply_pointer_patches_async(const std::vector<PointerPatch> & patches,
                                 StreamHandle stream);

}  // namespace streamllm_ext
