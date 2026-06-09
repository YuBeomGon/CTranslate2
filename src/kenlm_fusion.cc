#include "ctranslate2/kenlm_fusion.h"

#include <cstring>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#ifdef _WIN32
#  include <windows.h>
#else
#  include <stdlib.h>
#endif

#ifdef CT2_WITH_KENLM
#  include <lm/model.hh>
#  include <lm/virtual_interface.hh>
#endif

namespace ctranslate2 {

#ifdef CT2_WITH_KENLM
  namespace {

    constexpr float kLog10ToLn = 2.302585092994046f;

    struct KenlmBpeScorerCacheKey {
      std::string model_path;
      size_t text_token_limit;

      bool operator==(const KenlmBpeScorerCacheKey& other) const {
        return model_path == other.model_path && text_token_limit == other.text_token_limit;
      }
    };

    struct KenlmBpeScorerCacheKeyHash {
      size_t operator()(const KenlmBpeScorerCacheKey& key) const {
        const size_t path_hash = std::hash<std::string>()(key.model_path);
        const size_t limit_hash = std::hash<size_t>()(key.text_token_limit);
        return path_hash ^ (limit_hash + 0x9e3779b9 + (path_hash << 6) + (path_hash >> 2));
      }
    };

    class KenlmBpeStateBatch final : public LmStateBatch {
    public:
      KenlmBpeStateBatch(size_t size, size_t state_size)
        : _state_size(state_size)
        , _states(size * state_size)
      {
      }

      size_t size() const override {
        return _state_size == 0 ? 0 : _states.size() / _state_size;
      }

      void resize(size_t size) override {
        _states.resize(size * _state_size);
      }

      std::unique_ptr<LmStateBatch> clone_empty(size_t size) const override {
        return std::make_unique<KenlmBpeStateBatch>(size, _state_size);
      }

      void* state(size_t index) {
        return _states.data() + index * _state_size;
      }

      const void* state(size_t index) const {
        return _states.data() + index * _state_size;
      }

      size_t state_size() const {
        return _state_size;
      }

    private:
      size_t _state_size;
      std::vector<char> _states;
    };

    class KenlmBpeScorer final : public LmFusionScorer {
    public:
      KenlmBpeScorer(const std::string& model_path, size_t text_token_limit)
        : _model(lm::ngram::LoadVirtual(model_path.c_str()))
        , _state_size(_model->StateSize())
        , _word_indices(text_token_limit)
      {
        for (size_t token_id = 0; token_id < text_token_limit; ++token_id)
          _word_indices[token_id] = _model->BaseVocabulary().Index("t" + std::to_string(token_id));
      }

      std::unique_ptr<LmStateBatch>
      make_initial_states(size_t size,
                          const std::vector<std::vector<size_t>>* initial_histories,
                          dim_t beam_size) const override {
        auto states = std::make_unique<KenlmBpeStateBatch>(size, _state_size);

        if (!initial_histories) {
          for (size_t i = 0; i < size; ++i)
            _model->BeginSentenceWrite(states->state(i));
          return states;
        }

        if (beam_size <= 0)
          throw std::invalid_argument("The LM fusion beam size must be > 0");

        const size_t beams = static_cast<size_t>(beam_size);
        if (initial_histories->size() * beams != size)
          throw std::invalid_argument("The LM fusion initial history shape does not match the beam state size");

        std::vector<char> seeded(_state_size);
        std::vector<char> next(_state_size);
        for (size_t batch = 0; batch < initial_histories->size(); ++batch) {
          _model->BeginSentenceWrite(seeded.data());
          for (const size_t token_id : initial_histories->at(batch)) {
            _model->BaseFullScore(seeded.data(), word_index(token_id), next.data());
            std::swap(seeded, next);
          }

          for (size_t beam = 0; beam < beams; ++beam)
            std::memcpy(states->state(batch * beams + beam), seeded.data(), _state_size);
        }

        return states;
      }

      float score_token(const LmStateBatch& in_states,
                        size_t in_index,
                        size_t original_token_id,
                        LmStateBatch& out_states,
                        size_t out_index) const override {
        const auto& src = checked_batch(in_states);
        auto& dst = checked_batch(out_states);
        const auto ret = _model->BaseFullScore(src.state(in_index),
                                               word_index(original_token_id),
                                               dst.state(out_index));
        return ret.prob * kLog10ToLn;
      }

      void copy_state(const LmStateBatch& in_states,
                      size_t in_index,
                      LmStateBatch& out_states,
                      size_t out_index) const override {
        const auto& src = checked_batch(in_states);
        auto& dst = checked_batch(out_states);
        std::memcpy(dst.state(out_index), src.state(in_index), _state_size);
      }

      void gather(const LmStateBatch& src,
                  const std::vector<int32_t>& indices,
                  LmStateBatch& dst) const override {
        const auto& s = checked_batch(src);
        auto& d = checked_batch(dst);
        d.resize(indices.size());
        for (size_t i = 0; i < indices.size(); ++i) {
          if (indices[i] < 0 || static_cast<size_t>(indices[i]) >= s.size())
            throw std::out_of_range("Invalid LM fusion gather index");
          std::memcpy(d.state(i), s.state(indices[i]), _state_size);
        }
      }

      void keep_batches(const LmStateBatch& src,
                        const std::vector<int32_t>& kept_batch_ids,
                        dim_t beam_size,
                        LmStateBatch& dst) const override {
        const auto& s = checked_batch(src);
        auto& d = checked_batch(dst);
        const size_t beams = static_cast<size_t>(beam_size);
        d.resize(kept_batch_ids.size() * beams);

        size_t out = 0;
        for (const int32_t batch_id : kept_batch_ids) {
          if (batch_id < 0 || (static_cast<size_t>(batch_id) + 1) * beams > s.size())
            throw std::out_of_range("Invalid LM fusion batch index");
          for (size_t beam = 0; beam < beams; ++beam)
            std::memcpy(d.state(out++), s.state(static_cast<size_t>(batch_id) * beams + beam), _state_size);
        }
      }

    private:
      const KenlmBpeStateBatch& checked_batch(const LmStateBatch& batch) const {
        const auto* kenlm_batch = dynamic_cast<const KenlmBpeStateBatch*>(&batch);
        if (!kenlm_batch || kenlm_batch->state_size() != _state_size)
          throw std::invalid_argument("Invalid LM fusion state batch type");
        return *kenlm_batch;
      }

      KenlmBpeStateBatch& checked_batch(LmStateBatch& batch) const {
        auto* kenlm_batch = dynamic_cast<KenlmBpeStateBatch*>(&batch);
        if (!kenlm_batch || kenlm_batch->state_size() != _state_size)
          throw std::invalid_argument("Invalid LM fusion state batch type");
        return *kenlm_batch;
      }

      lm::WordIndex word_index(size_t token_id) const {
        if (token_id >= _word_indices.size())
          throw std::out_of_range("LM fusion token id exceeds the KenLM token table");
        return _word_indices[token_id];
      }

      std::unique_ptr<lm::base::Model> _model;
      size_t _state_size;
      std::vector<lm::WordIndex> _word_indices;
    };

    std::string canonical_model_path(const std::string& model_path) {
#ifdef _WIN32
      int wcount = MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, nullptr, 0);
      if (wcount == 0)
        throw std::runtime_error("Failed to resolve KenLM model path: " + model_path);
      std::wstring wpath(wcount, 0);
      MultiByteToWideChar(CP_UTF8, 0, model_path.c_str(), -1, &wpath[0], wcount);

      const DWORD full_size = GetFullPathNameW(wpath.c_str(), 0, nullptr, nullptr);
      if (full_size == 0)
        throw std::runtime_error("Failed to resolve KenLM model path: " + model_path);
      std::wstring full_path(full_size, 0);
      const DWORD written = GetFullPathNameW(wpath.c_str(), full_size, &full_path[0], nullptr);
      if (written == 0 || written >= full_size)
        throw std::runtime_error("Failed to resolve KenLM model path: " + model_path);
      full_path.resize(written);

      int count = WideCharToMultiByte(CP_UTF8, 0, full_path.c_str(), -1, nullptr, 0, nullptr, nullptr);
      if (count == 0)
        throw std::runtime_error("Failed to resolve KenLM model path: " + model_path);
      std::string path(count, 0);
      WideCharToMultiByte(CP_UTF8, 0, full_path.c_str(), -1, &path[0], count, nullptr, nullptr);
      if (!path.empty() && path.back() == '\0')
        path.pop_back();
      return path;
#else
      char* resolved = realpath(model_path.c_str(), nullptr);
      if (!resolved)
        throw std::runtime_error("Failed to resolve KenLM model path: " + model_path);
      std::string path(resolved);
      free(resolved);
      return path;
#endif
    }

    KenlmBpeScorerCacheKey
    make_cache_key(const std::string& model_path, size_t text_token_limit) {
      return {canonical_model_path(model_path), text_token_limit};
    }

  }
#endif

  std::shared_ptr<const LmFusionScorer>
  load_kenlm_bpe_scorer(const std::string& model_path, size_t text_token_limit) {
#ifdef CT2_WITH_KENLM
    static std::mutex cache_mutex;
    static std::unordered_map<KenlmBpeScorerCacheKey,
                              std::shared_ptr<const LmFusionScorer>,
                              KenlmBpeScorerCacheKeyHash> cache;

    const KenlmBpeScorerCacheKey key = make_cache_key(model_path, text_token_limit);
    const std::lock_guard<std::mutex> lock(cache_mutex);
    const auto it = cache.find(key);
    if (it != cache.end())
      return it->second;

    auto scorer = std::make_shared<KenlmBpeScorer>(key.model_path, text_token_limit);
    cache.emplace(key, scorer);
    return scorer;
#else
    (void)model_path;
    (void)text_token_limit;
    throw std::runtime_error("KenLM fusion requires CTranslate2 built with WITH_KENLM=ON");
#endif
  }

}
