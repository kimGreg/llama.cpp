#pragma once

#include "chunked_tensor.h"

namespace dp_moe_ext {

void apply_pointer_patches_async(const std::vector<PointerPatch> & patches,
                                 StreamHandle stream);

}  // namespace dp_moe_ext
