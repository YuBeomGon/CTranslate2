#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <unordered_map>

#include "ops/tile.h"
#include "storage_view.h"

namespace ctranslate2 {

  inline void split_batch_beam(StorageView& input, dim_t beam_size) {
    Shape shape = input.shape();
    shape.insert(shape.begin() + 1, beam_size);
    shape[0] /= beam_size;
    input.reshape(std::move(shape));
  }

  inline void merge_batch_beam(StorageView& input) {
    Shape shape = input.shape();
    shape[0] *= shape[1];
    shape.erase(shape.begin() + 1);
    input.reshape(std::move(shape));
  }

  inline void repeat_batch(StorageView& input, dim_t repeats) {
    input.expand_dims(1);
    ops::Tile(/*axis=*/1, repeats)(input);
    merge_batch_beam(input);
  }

  inline bool is_eos(const size_t id, const std::vector<size_t>& end_ids) {
    return std::find(end_ids.begin(), end_ids.end(), id) != end_ids.end();
  }

  // Helper class to disable tokens in the model output.
  class DisableTokens {
  public:
    DisableTokens(StorageView& logits,
                  const float disable_value = std::numeric_limits<float>::lowest());

    void add(dim_t batch_id, dim_t token_id) {
      const auto flat_index = batch_id * _vocabulary_size + token_id;

      if (_logits_data) {
        // On CPU we directly assign the value.
        _logits_data[flat_index] = _disable_value;

      } else {
        // On GPU we prepare a list of unique index to disable.
        const auto it = std::lower_bound(_flat_indices.begin(), _flat_indices.end(), flat_index);
        if (it == _flat_indices.end() || *it != flat_index)
          _flat_indices.insert(it, flat_index);
      }
    }

    // Disable a token for all batches.
    void add(dim_t token_id) {
      for (dim_t batch_id = 0; batch_id < _batch_size; ++batch_id)
        add(batch_id, token_id);
    }

    void apply();

  private:
    StorageView& _logits;
    float* _logits_data;
    const float _disable_value;
    const dim_t _batch_size;
    const dim_t _vocabulary_size;
    std::vector<int32_t> _flat_indices;
  };

  // Base class for processing the output logits.
  class LogitsProcessor {
  public:
    virtual ~LogitsProcessor() = default;

    virtual bool apply_first() const {
      return false;
    }

    virtual void apply(dim_t step,
                       StorageView& logits,
                       DisableTokens& disable_tokens,
                       const StorageView& sequences,
                       const std::vector<dim_t>& batch_offset,
                       const std::vector<std::vector<size_t>>* prefix) = 0;

  protected:
    dim_t get_batch_index(const dim_t batch_size,
                          const dim_t batch_id,
                          const std::vector<dim_t>& batch_offset) const {
      const auto beam_size = batch_size / batch_offset.size();
      return batch_offset[batch_id / beam_size];
    }

    dim_t get_sample_begin(const dim_t batch_size,
                           const dim_t batch_id,
                           const std::vector<dim_t>& batch_offset,
                           const std::vector<std::vector<size_t>>* prefix) const {
      return prefix ? prefix->at(get_batch_index(batch_size, batch_id, batch_offset)).size() : 0;
    }
  };

  // Apply a penalty to the score of previously generated tokens.
  class RepetitionPenalty : public LogitsProcessor {
  public:
    RepetitionPenalty(const float penalty);
    void apply(dim_t step,
               StorageView& logits,
               DisableTokens& disable_tokens,
               const StorageView& sequences,
               const std::vector<dim_t>& batch_offset,
               const std::vector<std::vector<size_t>>* prefix) override;

  private:
    const float _penalty;
  };

  // Prevent repetitions of ngrans with a specific size.
  class NoRepeatNgram : public LogitsProcessor {
  public:
    NoRepeatNgram(const size_t ngram_size);
    void apply(dim_t step,
               StorageView& logits,
               DisableTokens& disable_tokens,
               const StorageView& sequences,
               const std::vector<dim_t>& batch_offset,
               const std::vector<std::vector<size_t>>* prefix) override;

  private:
    const dim_t _ngram_size;
  };

  // Low-level, model-agnostic phrase bias path. step_bias is precomputed (= total_bias/(len-1), clamped).
  struct PhraseBiasEntry {
    std::vector<size_t> ids;
    float step_bias = 0.f;
    uint16_t min_prefix_len = 1;
  };

  // Reverse-prefix trie: maps a reversed suffix to the next token(s) to boost.
  class PhraseBiasTrie {
  public:
    void add(const PhraseBiasEntry& entry);
    // out[token] += step for every continuation rule whose prefix matches the suffix of `tail`.
    void lookup(const int32_t* tail, dim_t tail_len, std::map<size_t, float>& out) const;

  private:
    struct Node {
      std::unordered_map<size_t, Node> children;     // key = previous token
      std::vector<std::pair<size_t, float>> actions; // (next_token, step) valid at this depth
    };
    Node _root;
  };

  // Build an immutable, shareable trie from entries (call once at load time).
  std::shared_ptr<const PhraseBiasTrie>
  build_phrase_bias_trie(const std::vector<PhraseBiasEntry>& entries);

  // Positive continuation phrase bias. apply_first()=false so it runs after no-speech.
  class PhraseBiasProcessor : public LogitsProcessor {
  public:
    // Convenience: build a fresh trie from entries.
    explicit PhraseBiasProcessor(const std::vector<PhraseBiasEntry>& entries,
                                 float max_token_delta = 1.0f);
    // Share a pre-built trie (no rebuild) — used by the load-time cached path.
    explicit PhraseBiasProcessor(std::shared_ptr<const PhraseBiasTrie> trie,
                                 float max_token_delta = 1.0f);

    bool apply_first() const override { return false; }

    void apply(dim_t step,
               StorageView& logits,
               DisableTokens& disable_tokens,
               const StorageView& sequences,
               const std::vector<dim_t>& batch_offset,
               const std::vector<std::vector<size_t>>* prefix) override;

  private:
    std::shared_ptr<const PhraseBiasTrie> _trie;
    float _max_token_delta;
  };

  // Disable the generation of some sequences of tokens.
  class SuppressSequences : public LogitsProcessor {
  public:
    SuppressSequences(std::vector<std::vector<size_t>> sequences);
    void apply(dim_t step,
               StorageView& logits,
               DisableTokens& disable_tokens,
               const StorageView& sequences,
               const std::vector<dim_t>& batch_offset,
               const std::vector<std::vector<size_t>>* prefix) override;

  private:
    std::vector<size_t> _ids;
    std::vector<std::vector<size_t>> _sequences;
  };

  // Disable the generation of some tokens.
  class SuppressTokens : public LogitsProcessor {
  public:
    SuppressTokens(std::vector<size_t> ids);
    void apply(dim_t step,
               StorageView& logits,
               DisableTokens& disable_tokens,
               const StorageView& sequences,
               const std::vector<dim_t>& batch_offset,
               const std::vector<std::vector<size_t>>* prefix) override;

  private:
    const std::vector<size_t> _ids;
  };

  // Disable the generation of some tokens at the first unconstrained decoding step.
  class SuppressTokensBegin : public LogitsProcessor {
  public:
    SuppressTokensBegin(std::vector<size_t> ids);
    void apply(dim_t step,
               StorageView& logits,
               DisableTokens& disable_tokens,
               const StorageView& sequences,
               const std::vector<dim_t>& batch_offset,
               const std::vector<std::vector<size_t>>* prefix) override;

  private:
    const std::vector<size_t> _ids;
  };

}
