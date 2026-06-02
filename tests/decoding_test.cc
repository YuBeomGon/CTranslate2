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
