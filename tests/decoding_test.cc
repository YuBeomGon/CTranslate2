#include <ctranslate2/decoding.h>

#include "test_utils.h"

namespace {

  class FakeLmStateBatch final : public LmStateBatch {
  public:
    explicit FakeLmStateBatch(size_t size = 0)
      : states(size, 0)
    {
    }

    size_t size() const override {
      return states.size();
    }

    void resize(size_t size) override {
      states.resize(size);
    }

    std::unique_ptr<LmStateBatch> clone_empty(size_t size) const override {
      return std::make_unique<FakeLmStateBatch>(size);
    }

    std::vector<int> states;
  };

  class FakeLmFusionScorer final : public LmFusionScorer {
  public:
    explicit FakeLmFusionScorer(bool prefer_inside_topk = true)
      : _prefer_inside_topk(prefer_inside_topk)
    {
    }

    std::unique_ptr<LmStateBatch> make_initial_states(size_t size) const override {
      return std::make_unique<FakeLmStateBatch>(size);
    }

    float score_token(const LmStateBatch& in_states,
                      size_t in_index,
                      size_t original_token_id,
                      LmStateBatch& out_states,
                      size_t out_index) const override {
      const auto& src = static_cast<const FakeLmStateBatch&>(in_states);
      auto& dst = static_cast<FakeLmStateBatch&>(out_states);
      const int history = src.states[in_index];
      dst.states[out_index] = history * 10 + static_cast<int>(original_token_id);

      if (_prefer_inside_topk && history == 0 && original_token_id == 2)
        return 10;
      if (!_prefer_inside_topk && history == 0 && original_token_id == 3)
        return 10;
      if (history == 2 && original_token_id == 3)
        return 10;
      if (history == 1 && original_token_id == 4)
        return 10;
      return 0;
    }

    void copy_state(const LmStateBatch& in_states,
                    size_t in_index,
                    LmStateBatch& out_states,
                    size_t out_index) const override {
      const auto& src = static_cast<const FakeLmStateBatch&>(in_states);
      auto& dst = static_cast<FakeLmStateBatch&>(out_states);
      dst.states[out_index] = src.states[in_index];
    }

    void gather(const LmStateBatch& src,
                const std::vector<int32_t>& indices,
                LmStateBatch& dst) const override {
      const auto& s = static_cast<const FakeLmStateBatch&>(src);
      auto& d = static_cast<FakeLmStateBatch&>(dst);
      d.resize(indices.size());
      for (size_t i = 0; i < indices.size(); ++i)
        d.states[i] = s.states[indices[i]];
    }

    void keep_batches(const LmStateBatch& src,
                      const std::vector<int32_t>& kept_batch_ids,
                      dim_t beam_size,
                      LmStateBatch& dst) const override {
      const auto& s = static_cast<const FakeLmStateBatch&>(src);
      auto& d = static_cast<FakeLmStateBatch&>(dst);
      d.resize(kept_batch_ids.size() * beam_size);
      size_t out = 0;
      for (const int32_t batch_id : kept_batch_ids) {
        for (dim_t beam = 0; beam < beam_size; ++beam)
          d.states[out++] = s.states[batch_id * beam_size + beam];
      }
    }

  private:
    bool _prefer_inside_topk;
  };

  class ScriptedDecoder final : public layers::Decoder {
  public:
    explicit ScriptedDecoder(std::vector<std::vector<float>> step_logits)
      : layers::Decoder(Device::CPU)
      , _step_logits(std::move(step_logits))
    {
    }

    layers::DecoderState initial_state(bool iterative_decoding = true) const override {
      (void)iterative_decoding;
      return {{"state", StorageView({1}, std::vector<int32_t>{0})}};
    }

    void operator()(dim_t step,
                    const StorageView& ids,
                    layers::DecoderState& state,
                    StorageView* logits = nullptr,
                    StorageView* attention = nullptr) override {
      (void)state;
      (void)attention;
      const dim_t rows = ids.dim(0);
      logits->resize({rows, output_size()});
      const auto& scores = _step_logits[std::min<size_t>(step, _step_logits.size() - 1)];
      for (dim_t row = 0; row < rows; ++row) {
        for (dim_t id = 0; id < output_size(); ++id)
          logits->at<float>({row, id}) = scores[id];
      }
    }

    void operator()(const StorageView& ids,
                    const StorageView& lengths,
                    layers::DecoderState& state,
                    StorageView& logits,
                    StorageView* attention = nullptr) override {
      (void)ids;
      (void)lengths;
      (void)state;
      (void)logits;
      (void)attention;
      throw std::runtime_error("not implemented");
    }

    DataType output_type() const override {
      return DataType::FLOAT32;
    }

    dim_t output_size() const override {
      return _step_logits.front().size();
    }

  protected:
    layers::Dense& output_layer() override {
      throw std::runtime_error("not implemented");
    }

  private:
    std::vector<std::vector<float>> _step_logits;
  };

  DecodingOptions make_lm_fusion_options(bool prefer_inside_topk = true) {
    DecodingOptions options;
    options.beam_size = 2;
    options.max_length = 2;
    options.lm_fusion.alpha = 1;
    options.lm_fusion.asr_topk = 2;
    options.lm_fusion.text_token_limit = 5;
    options.lm_fusion_scorer = std::make_shared<FakeLmFusionScorer>(prefer_inside_topk);
    return options;
  }

}

TEST(DecodingTest, DisableTokens) {
  StorageView input({2, 5}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  StorageView expected({2, 5}, std::vector<float>{1, 0, 0, 4, 5, 6, 7, 0, 9, 0});

  DisableTokens disable_tokens(input, 0);
  disable_tokens.add(2);
  disable_tokens.add(0, 1);
  disable_tokens.add(1, 4);
  disable_tokens.apply();

  expect_storage_eq(input, expected);
}

TEST(DecodingTest, LmFusionSelectsCandidateInsideAsrTopK) {
  ScriptedDecoder decoder({
    {0, 5, 4, -10, -20},
    {0, -10, -10, 4, 5},
  });
  auto state = decoder.initial_state();
  auto options = make_lm_fusion_options();

  const auto results = decode(decoder, state, {{0}}, {4}, options);

  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results[0].hypotheses.size(), 1);
  EXPECT_EQ(results[0].hypotheses[0], (std::vector<size_t>{2, 3}));
}

TEST(DecodingTest, LmFusionDoesNotRescueOutsideAsrTopK) {
  ScriptedDecoder decoder({
    {0, 5, 4, -10, -20},
    {0, -10, -10, 4, 5},
  });
  auto state = decoder.initial_state();
  auto options = make_lm_fusion_options(false);

  const auto results = decode(decoder, state, {{0}}, {4}, options);

  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results[0].hypotheses.size(), 1);
  EXPECT_EQ(results[0].hypotheses[0], (std::vector<size_t>{1, 4}));
}
