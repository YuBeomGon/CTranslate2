#include <ctranslate2/kenlm_fusion.h>

#include <cmath>
#include <cstdlib>
#include <stdexcept>

#include <gtest/gtest.h>

namespace {

TEST(KenlmFusionTest, MissingBuildThrowsClearError) {
#ifdef CT2_WITH_KENLM
  GTEST_SKIP() << "Only applies to WITH_KENLM=OFF builds";
#else
  try {
    ctranslate2::load_kenlm_bpe_scorer("missing.binary", 4);
    FAIL() << "Expected an exception";
  } catch (const std::runtime_error& e) {
    EXPECT_STREQ(e.what(), "KenLM fusion requires CTranslate2 built with WITH_KENLM=ON");
  }
#endif
}

TEST(KenlmFusionTest, FixtureSmoke) {
#ifndef CT2_WITH_KENLM
  GTEST_SKIP() << "Requires WITH_KENLM=ON";
#else
  const char* model_path = std::getenv("CT2_KENLM_TEST_BINARY");
  if (!model_path)
    GTEST_SKIP() << "Set CT2_KENLM_TEST_BINARY to run the KenLM scorer smoke test";

  const auto scorer = ctranslate2::load_kenlm_bpe_scorer(model_path, 8);
  const std::vector<std::vector<size_t>> histories{{1, 2}};
  auto states = scorer->make_initial_states(2, &histories, 2);
  auto out_states = states->clone_empty(1);
  const float score = scorer->score_token(*states, 0, 3, *out_states, 0);
  EXPECT_TRUE(std::isfinite(score));
#endif
}

}
