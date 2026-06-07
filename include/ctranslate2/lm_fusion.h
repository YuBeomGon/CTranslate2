#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "ctranslate2/types.h"

namespace ctranslate2 {

  struct LmFusionOptions {
    float alpha = 0;
    size_t asr_topk = 50;
    size_t text_token_limit = 0;
    bool debug = false;
  };

  class LmStateBatch {
  public:
    virtual ~LmStateBatch() = default;

    virtual size_t size() const = 0;
    virtual void resize(size_t size) = 0;
    virtual std::unique_ptr<LmStateBatch> clone_empty(size_t size) const = 0;
  };

  class LmFusionScorer {
  public:
    virtual ~LmFusionScorer() = default;

    virtual std::unique_ptr<LmStateBatch>
    make_initial_states(size_t size,
                        const std::vector<std::vector<size_t>>* initial_histories = nullptr,
                        dim_t beam_size = 1) const = 0;

    virtual float score_token(const LmStateBatch& in_states,
                              size_t in_index,
                              size_t original_token_id,
                              LmStateBatch& out_states,
                              size_t out_index) const = 0;

    virtual void copy_state(const LmStateBatch& in_states,
                            size_t in_index,
                            LmStateBatch& out_states,
                            size_t out_index) const = 0;

    virtual void gather(const LmStateBatch& src,
                        const std::vector<int32_t>& indices,
                        LmStateBatch& dst) const = 0;

    virtual void keep_batches(const LmStateBatch& src,
                              const std::vector<int32_t>& kept_batch_ids,
                              dim_t beam_size,
                              LmStateBatch& dst) const = 0;
  };

}
