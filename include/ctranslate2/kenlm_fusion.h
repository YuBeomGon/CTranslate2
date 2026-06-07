#pragma once

#include <memory>
#include <string>

#include "ctranslate2/lm_fusion.h"

namespace ctranslate2 {

  std::shared_ptr<const LmFusionScorer>
  load_kenlm_bpe_scorer(const std::string& model_path, size_t text_token_limit);

}
