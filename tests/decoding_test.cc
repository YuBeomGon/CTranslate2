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
