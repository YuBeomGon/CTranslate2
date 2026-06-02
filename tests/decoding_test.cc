#include <ctranslate2/decoding.h>
#include <ctranslate2/models/whisper.h>

#include "test_utils.h"

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

TEST(PhraseBiasTest, WhisperOptionsDefaultsEmpty) {
  ctranslate2::models::WhisperOptions options;
  EXPECT_TRUE(options.phrase_biases.empty());
  EXPECT_EQ(options.compiled_phrase_bias_trie, nullptr);
}

TEST(PhraseBiasTest, PhraseBiasPathDefaults) {
  ctranslate2::models::PhraseBiasPath path;
  EXPECT_EQ(path.ids.size(), 0u);
  EXPECT_FLOAT_EQ(path.start_bias, 0.f);
  EXPECT_FLOAT_EQ(path.step_bias, 0.f);
  EXPECT_EQ(path.min_prefix_len, 1u);
  EXPECT_EQ(path.mode, ctranslate2::models::PhraseBiasMode::Soft);
}

TEST(PhraseBiasTest, TrieSharedPrefixReturnsBoth) {
  PhraseBiasTrie trie;
  trie.add(PhraseBiasEntry{{1, 2, 3}, 0.25f, 1});  // [A=1,B=2,C=3]
  trie.add(PhraseBiasEntry{{1, 2, 4}, 0.25f, 1});  // [A=1,B=2,D=4]
  trie.add(PhraseBiasEntry{{8, 9}, 0.5f, 1});      // [X=8,Y=9]

  auto boost = [&](std::vector<int32_t> tail) {
    std::map<size_t, float> out;
    trie.lookup(tail.data(), static_cast<dim_t>(tail.size()), out);
    return out;
  };

  EXPECT_EQ(boost({1}), (std::map<size_t,float>{{2, 0.5f}}));        // [A] -> B (두 path 합산 0.25+0.25)
  EXPECT_EQ(boost({1, 2}), (std::map<size_t,float>{{3, 0.25f}, {4, 0.25f}}));  // [A,B] -> C,D
  EXPECT_EQ(boost({8}), (std::map<size_t,float>{{9, 0.5f}}));        // [X] -> Y
  EXPECT_TRUE(boost({7}).empty());                                  // 불일치
  EXPECT_TRUE(boost({}).empty());                                   // suffix 없음
}

TEST(PhraseBiasTest, TrieMinPrefixLenSkipsShortMatch) {
  PhraseBiasTrie trie;
  trie.add(PhraseBiasEntry{{1, 2, 3}, 0.25f, 2});  // min_prefix_len=2
  std::map<size_t, float> out1;
  trie.lookup(std::vector<int32_t>{1}.data(), 1, out1);
  EXPECT_TRUE(out1.empty());                        // [A]만으론 boost 없음
  std::map<size_t, float> out2;
  trie.lookup(std::vector<int32_t>{1, 2}.data(), 2, out2);
  EXPECT_EQ(out2, (std::map<size_t,float>{{3, 0.25f}}));  // [A,B] -> C
}

TEST(PhraseBiasTest, TrieSkipsTooShortEntry) {
  PhraseBiasTrie trie;
  trie.add(PhraseBiasEntry{{5}, 0.5f, 1});  // 1-token -> continuation 불가 -> skip
  std::map<size_t, float> out;
  trie.lookup(std::vector<int32_t>{5}.data(), 1, out);
  EXPECT_TRUE(out.empty());
}

TEST(PhraseBiasTest, ProcessorBoostsNextTokenOnSuffixMatch) {
  // vocab=5, batch=1. logits 모두 0.
  StorageView logits({1, 5}, std::vector<float>(5, 0.f));
  DisableTokens disable(logits, std::numeric_limits<float>::lowest());

  std::vector<PhraseBiasEntry> entries = {{{1, 2, 3}, 0.25f, 1}};  // [A=1,B=2,C=3]
  PhraseBiasProcessor proc(entries);

  StorageView seq({1, 1}, std::vector<int32_t>{1});  // suffix [A]
  proc.apply(1, logits, disable, seq, {0}, nullptr);

  StorageView expected({1, 5}, std::vector<float>{0, 0, 0.25f, 0, 0});  // token 2(B) += 0.25
  expect_storage_eq(logits, expected);
}

TEST(PhraseBiasTest, ProcessorOverlapSumsAndClamps) {
  StorageView logits({1, 5}, std::vector<float>(5, 0.f));
  DisableTokens disable(logits, std::numeric_limits<float>::lowest());
  // 두 phrase 모두 [A]->B boost: 0.6 + 0.6 = 1.2 -> clamp 1.0
  std::vector<PhraseBiasEntry> entries = {{{1, 2}, 0.6f, 1}, {{1, 2}, 0.6f, 1}};
  PhraseBiasProcessor proc(entries, /*max_token_delta=*/1.0f);
  StorageView seq({1, 1}, std::vector<int32_t>{1});
  proc.apply(1, logits, disable, seq, {0}, nullptr);
  StorageView expected({1, 5}, std::vector<float>{0, 0, 1.0f, 0, 0});
  expect_storage_eq(logits, expected);
}

TEST(PhraseBiasTest, ProcessorNoMatchAndNullAreNoOp) {
  std::vector<PhraseBiasEntry> entries = {{{1, 2, 3}, 0.25f, 1}};
  PhraseBiasProcessor proc(entries);
  // 불일치
  StorageView logits1({1, 5}, std::vector<float>(5, 0.f));
  DisableTokens d1(logits1, std::numeric_limits<float>::lowest());
  StorageView seq({1, 1}, std::vector<int32_t>{7});
  proc.apply(1, logits1, d1, seq, {0}, nullptr);
  expect_storage_eq(logits1, StorageView({1, 5}, std::vector<float>(5, 0.f)));
  // null sequences (step 0)
  StorageView logits2({1, 5}, std::vector<float>(5, 0.f));
  DisableTokens d2(logits2, std::numeric_limits<float>::lowest());
  StorageView empty;
  proc.apply(0, logits2, d2, empty, {0}, nullptr);
  expect_storage_eq(logits2, StorageView({1, 5}, std::vector<float>(5, 0.f)));
}

TEST(PhraseBiasTest, ConvertModelOptionToEntries) {
  using namespace ctranslate2::models;
  std::vector<PhraseBias> biases(1);
  PhraseBiasPath p;
  p.ids = {1, 2, 3};
  p.step_bias = 0.25f;
  p.min_prefix_len = 1;
  biases[0].token_paths.push_back(p);

  std::vector<ctranslate2::PhraseBiasEntry> entries = to_phrase_bias_entries(biases);
  ASSERT_EQ(entries.size(), 1u);
  EXPECT_EQ(entries[0].ids, (std::vector<size_t>{1, 2, 3}));
  EXPECT_FLOAT_EQ(entries[0].step_bias, 0.25f);
  EXPECT_EQ(entries[0].min_prefix_len, 1u);
}

TEST(PhraseBiasTest, SharedTrieReusedByProcessors) {
  std::vector<PhraseBiasEntry> entries = {{{1, 2, 3}, 0.25f, 1}};
  std::shared_ptr<const PhraseBiasTrie> trie = build_phrase_bias_trie(entries);
  ASSERT_TRUE(trie != nullptr);

  for (int i = 0; i < 2; ++i) {
    StorageView logits({1, 5}, std::vector<float>(5, 0.f));
    DisableTokens disable(logits, std::numeric_limits<float>::lowest());
    PhraseBiasProcessor proc(trie);
    StorageView seq({1, 1}, std::vector<int32_t>{1});
    proc.apply(1, logits, disable, seq, {0}, nullptr);
    StorageView expected({1, 5}, std::vector<float>{0, 0, 0.25f, 0, 0});
    expect_storage_eq(logits, expected);
  }
}

// CPU/GPU parity: logits는 device/dtype, sequences는 host(int32) — 실제 generate 계약과 동일.
class PhraseBiasProcessorFPTest : public ::testing::TestWithParam<FloatType> {
};

TEST_P(PhraseBiasProcessorFPTest, CpuGpuParity) {
  const Device device = GetParam().device;
  const DataType dtype = GetParam().dtype;
  const float error = GetParam().error;

  // batch=2 (row-wise batch>1 커버; beam end-to-end 아님), vocab=6, logits 모두 0.
  StorageView logits({2, 6}, std::vector<float>(12, 0.f), device);
  logits = logits.to(dtype);
  DisableTokens disable(logits, std::numeric_limits<float>::lowest());

  // shared prefix [1,2,*] 두 path → overlap 합산 검증.
  std::vector<PhraseBiasEntry> entries = {
    {{1, 2, 3}, 0.3f, 1},
    {{1, 2, 4}, 0.3f, 1},
  };
  PhraseBiasProcessor proc(entries);

  // sequences는 항상 host(CPU). row0 suffix [...,1] → token2 boost(0.3+0.3=0.6).
  //                              row1 suffix [1,2]  → token3,4 boost(각 0.3).
  StorageView seq({2, 2}, std::vector<int32_t>{0, 1,  1, 2});
  proc.apply(2, logits, disable, seq, {0}, nullptr);

  StorageView expected({2, 6}, std::vector<float>{
      0, 0, 0.6f, 0,    0,    0,    // row0: token2 += 0.6 (합산)
      0, 0, 0,    0.3f, 0.3f, 0},   // row1: token3,4 += 0.3
      Device::CPU);
  expect_storage_eq(logits, expected.to(device).to(dtype), error);
}

INSTANTIATE_TEST_SUITE_P(CPU, PhraseBiasProcessorFPTest,
                         ::testing::Values(FloatType{Device::CPU, DataType::FLOAT32, 1e-5}),
                         fp_test_name);
#ifdef CT2_WITH_CUDA
INSTANTIATE_TEST_SUITE_P(CUDA, PhraseBiasProcessorFPTest,
                         ::testing::Values(FloatType{Device::CUDA, DataType::FLOAT32, 1e-5},
                                           FloatType{Device::CUDA, DataType::FLOAT16, 1e-2},
                                           FloatType{Device::CUDA, DataType::BFLOAT16, 4e-2}),
                         fp_test_name);
#endif
