
#include <vector>
#include <cstdint>

static const int64_t vocabSize { 32 };
static const int64_t hiddenSize { 768 };
static const int64_t nHiddenLayers { 12 };
static const int64_t nAttentionHeads { 12 };
static const int64_t nConvPosEmbeddings { 128 };
static const int64_t nConvPosEmbeddingGroups { 16 };
static const int64_t intermediateSize { 3072 };
static const std::vector<int64_t> convDimensions { 512, 512, 512, 512, 512, 512, 512 };
static const std::vector<int64_t> convStrides { 5, 2, 2, 2, 2, 2, 2 };
static const std::vector<int64_t> convKernels { 10, 3, 3, 3, 3, 2, 2 };

