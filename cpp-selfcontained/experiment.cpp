
#include <iostream>
#include <fstream>
#include <cmath>
#include <memory>

#include <sndfile.h>

#include "withoutlib.cpp"

#include "../data/weights.hpp"

using namespace std;

static const int64_t hiddenSize { 768 };
static const int64_t nHiddenLayers { 12 };
static const int64_t nAttentionHeads { 12 };
static const int64_t nConvPosEmbeddings { 128 };
static const int64_t nConvPosEmbeddingGroups { 16 };
static const vector<int64_t> convDimensions { 512, 512, 512, 512, 512, 512, 512 };
static const vector<int64_t> convStrides { 5, 2, 2, 2, 2, 2, 2 };
static const vector<int64_t> convKernels { 10, 3, 3, 3, 3, 2, 2 };

#define DEBUG_TENSOR_SHAPES 1
#ifdef DEBUG_TENSOR_SHAPES
#include <cxxabi.h>
#endif

void dump(const localnn::Tensor &tt, string filebase)
{
    string filename = filebase + ".csv";
    ofstream csv(filename);

    
    int base = tt.sizes.size() - 2;
    int nrows = 1;
    
    if (base < -1 || base > 2) {
        cerr << "unsupported shape in dump";
        exit(2);
    }
    if (base >= 0) {
        nrows = tt.sizes[base + 0];
    }

    int ncols = tt.sizes[base + 1];

    cerr << "writing " << nrows << "-row " << ncols << "-column csv to "
         << filename << endl;
    
    for (int j = 0; j < ncols; ++j) {
        csv << j;
        if (j + 1 < ncols) {
            csv << ",";
        } else {
            csv << endl;
        }
    }
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            switch (base) {
            case -1: csv << tt.at(j); break;
            case 0: csv << tt.at(i, j); break;
            case 1: csv << tt.at(0, i, j); break;
            case 2: csv << tt.at(0, 0, i, j); break;
            }
                
            if (j + 1 < ncols) {
                csv << ",";
            } else {
                csv << endl;
            }
        }
    }
}

void localLinearImpl(const localnn::Tensor &in, int64_t inbase,
                     localnn::Tensor &out, int64_t outbase,
                     const localnn::Tensor &weight,
                     const localnn::Tensor &bias,
                     int64_t rank,
                     int64_t rix)
{
    const float *indata = in.data();
    float *outdata = out.mutableData();
    if (rix + 1 == rank) {
        const int64_t insize = in.sizes[rix];
        const int64_t instride = in.strides[rix];
        const int64_t outsize = out.sizes[rix];
        const int64_t outstride = out.strides[rix];
#pragma omp parallel for
        for (int64_t j = 0; j < outsize; ++j) {
            float x = 0.f;
#pragma GCC ivdep
            for (int64_t i = 0; i < insize; ++i) {
                x += weight.at(j, i) * indata[inbase + i * instride];
            }
            x += bias.at(j);
            outdata[outbase + j * outstride] = x;
        }
    } else {
#pragma omp parallel for
        for (int64_t i = 0; i < in.sizes[rix]; ++i) {
            localLinearImpl(in, inbase + i * in.strides[rix],
                            out, outbase + i * out.strides[rix],
                            weight, bias,
                            rank, rix + 1);
        }
    }
}

localnn::Tensor localLinear_(const localnn::Tensor &in,
                             const localnn::Tensor &weight,
                             const localnn::Tensor &bias)
{
    auto rank = in.rank;
    auto outsizes = in.sizes;
    outsizes[rank-1] = weight.sizes[0];
    auto out = localnn::Tensor(outsizes);
    int rix = 0;
    while (in.sizes[rix] == 1) ++rix;
    localLinearImpl(in, 0,
                    out, 0,
                    weight, bias,
                    rank, rix);
    return out;
}

struct Module {
    virtual void prepare(string key) = 0;

    localnn::Tensor loadData(string keybase, string suffix) {
        string key = keybase;
        if (suffix != "") {
            key += "." + suffix;
        }
        if (key == "") {
            throw std::runtime_error("empty key");
        }
        if (key[0] == '.') {
            key = key.substr(1);
        }
        cerr << "Loading: " << key << " ... ";
        vector<int64_t> sizes;
        if (const float *data = lookup_model_data(key, sizes)) {
            cerr << "succeeded" << endl;
            return localnn::Tensor::fromConst(sizes, data);
        } else {
            cerr << "FAILED" << endl;
            throw std::runtime_error("failed to load data for key: " + key);
        }
    }
};

struct LayerBase : Module {
    virtual localnn::Tensor forward(const localnn::Tensor &x) = 0;
};

void localGelu_(localnn::Tensor &t)
{
    const double alpha = M_SQRT1_2;

    //!!! +omp

    float *tdata = t.mutableData();
    for (int64_t i = 0; i < t.numel(); ++i) {
        double x = tdata[i];
        x = x * 0.5 * (1.0 + std::erf(x * alpha));
        tdata[i] = float(x);
    }
}

// We always copy for reshape and transpose, because we can only
// handle contiguous layouts in some other functions

localnn::Tensor localReshape(const localnn::Tensor &t, vector<int64_t> outsizes)
{
    for (int i = 0; i < outsizes.size(); ++i) {
        if (outsizes[i] < 0) {
            outsizes[i] = t.numel();
            for (int j = 0; j < outsizes.size(); ++j) {
                if (j != i && outsizes[j] > 0) {
                    outsizes[i] /= outsizes[j];
                }
            }
            break;
        }
    }
    return localnn::Tensor(outsizes, t.copiedData());
}

localnn::Tensor localTranspose12of3(const localnn::Tensor &t)
{
    if (t.rank != 3) {
        throw std::runtime_error("shape");
    }
    int64_t a = t.sizes[0];
    int64_t b = t.sizes[1];
    int64_t c = t.sizes[2];
    vector<int64_t> outsizes = { a, c, b };
    localnn::Tensor out = localnn::Tensor(outsizes);
    const float *tdata = t.data();
    float *outdata = out.mutableData();
    for (int64_t i = 0; i < a; ++i) {
        for (int64_t j = 0; j < b; ++j) {
#pragma GCC ivdep
            for (int64_t k = 0; k < c; ++k) {
                outdata[out.index(i, k, j)] = tdata[t.index(i, j, k)];
            }
        }
    }
    return out;
}

localnn::Tensor localTranspose12of4(const localnn::Tensor &t)
{
    if (t.rank != 4) {
        throw std::runtime_error("shape");
    }
    int64_t a = t.sizes[0];
    int64_t b = t.sizes[1];
    int64_t c = t.sizes[2];
    int64_t d = t.sizes[3];
    vector<int64_t> outsizes = { a, c, b, d };
    localnn::Tensor out(outsizes);
    const float *tdata = t.data();
    float *outdata = out.mutableData();
    for (int64_t i = 0; i < a; ++i) {
        for (int64_t j = 0; j < b; ++j) {
            for (int64_t k = 0; k < c; ++k) {
#pragma GCC ivdep
                for (int64_t m = 0; m < d; ++m) {
                    outdata[out.index(i, k, j, m)] = tdata[t.index(i, j, k, m)];
                }
            }
        }
    }
    return out;
}

localnn::Tensor localConv1d_(const localnn::Tensor &t,
                             int64_t ch_in, int64_t ch_out, int64_t ksize,
                             int64_t stride, int64_t padding, int64_t groups,
                             const localnn::Tensor &w,
                             const localnn::Tensor *bp)
{
    if (t.rank != 3 || t.strides[2] != 1) {
        throw std::runtime_error("unsupported format for input");
    }
    if (w.rank != 3 || w.strides[2] != 1) {
        throw std::runtime_error("unsupported format for weight");
    }
    
    // batchsize * inchannels * inlength -> batchsize * outchannels * outlength

    int64_t l_in = t.sizes[2];
    int64_t l_out = (l_in + 2 * padding - (ksize - 1) - 1) / stride + 1;

    cerr << "l_in = " << l_in << ", l_out = " << l_out << endl;

    localnn::Tensor out({ t.sizes[0], ch_out, l_out });
    
    const float *bbase = nullptr;
    if (bp) {
        bbase = bp->data();
    }

    for (int64_t b = 0; b < t.sizes[0]; ++b) {
        for (int64_t g = 0; g < groups; ++g) {
            int c0 = g * (ch_out / groups);
            int k0 = g * (ch_in / groups);
            int g0 = g * ((out.numel() / t.sizes[0]) / groups);
#pragma omp parallel for
            for (int64_t c = 0; c < ch_out / groups; ++c) {

                float *const outbase =
                    out.mutableData() + g0 + out.index(b, c);

                for (int64_t k = 0; k < ch_in / groups; ++k) {
                    const float *const wbase =
                        w.data() + w.index(c + c0, k);
                    const float *const tbase =
                        t.data() + t.index(b, k + k0);
#pragma GCC ivdep
                    for (int64_t i = 0; i < ksize; ++i) {
                        if (padding == 0) {
                            for (int64_t x = 0; x < l_out; ++x) {
                                outbase[x] += wbase[i] * tbase[x * stride + i];
                            }
                        } else {
                            // ew
                            for (int64_t x = 0; x < l_out; ++x) {
                                int64_t x0 = x * stride + i - padding;
                                if (x0 < 0 || x0 >= l_in) continue;
                                outbase[x] += wbase[i] * tbase[x0];
                            }
                        }
                    }
                }

                if (bbase) {
#pragma GCC ivdep
                    for (int64_t x = 0; x < l_out; ++x) {
                        outbase[x] += bbase[c + c0];
                    }
                }
            }
        }
    }

    return out;
}

void localSoftmax_(localnn::Tensor &t)
{
    int64_t h = *t.sizes.rbegin();
    int64_t m = t.numel() / h;

    for (int64_t j = 0; j < m; ++j) {

        float *base = t.mutableData() + j * h;

        double sum = 0.0;
        for (int64_t i = 0; i < h; ++i) {
            double x = exp(base[i]);
            sum += x;
            base[i] = x;
        }
        if (sum != 0.0) {
            for (int64_t i = 0; i < h; ++i) {
                base[i] /= sum;
            }
        }
    }
}

void localLayerNorm_(localnn::Tensor &t,
                     const localnn::Tensor &weight, const localnn::Tensor &bias,
                     bool weightsPerInstance)
{
    int64_t h = *t.sizes.rbegin();
    int64_t m = t.numel() / h;

    for (int64_t j = 0; j < m; ++j) {

        float *base = t.mutableData() + j * h;
        
        double mean = 0.0;
        for (int64_t i = 0; i < h; ++i) {
            mean += base[i];
        }
        mean /= double(h);

        double variance = 0.0;
        for (int64_t i = 0; i < h; ++i) {
            variance += (base[i] - mean) * (base[i] - mean);
        }
        variance /= double(h);

        double eps = 1.0e-5;
        double sd = sqrt(variance + eps);
            
        for (int64_t i = 0; i < h; ++i) {

            double x = base[i];
            double y = (x - mean) / sd;

            if (weightsPerInstance) {
                // This case is for where the original model uses a
                // GroupNorm layer. The GroupNorm is supplied with
                // groups == channels so is actually performing an
                // instance norm
                y = y * weight.at(j) + bias.at(j);
            } else {
                y = y * weight.at(i) + bias.at(i);
            }                

            base[i] = y;
        }
    }
}

localnn::Tensor localBMM_(const localnn::Tensor &t, const localnn::Tensor &m)
{
    if (t.rank != 3 || m.rank != 3) {
        cerr << "unsupported rank (" << t.rank << " or "
             << m.rank << ", should both be 3)" << endl;
        throw std::runtime_error("shape");
    }        
    
    localnn::Tensor out({ t.sizes[0], t.sizes[1], m.sizes[2] });

    if (t.sizes[2] != m.sizes[1]) {
        cerr << "incompatible sizes (" << t.sizes[2] << " != "
             << m.sizes[1] << ")" << endl;
        throw std::runtime_error("shape");
    }

    float *outdata = out.mutableData();
    
    for (int64_t b = 0; b < t.sizes[0]; ++b) {
#pragma omp parallel for
        for (int64_t i = 0; i < t.sizes[1]; ++i) {
            for (int64_t j = 0; j < m.sizes[2]; ++j) {
                double d = 0.0;
#pragma GCC ivdep
                for (int64_t k = 0; k < m.sizes[1]; ++k) {
                    d += t.at(b, i, k) * m.at(b, k, j);
                }
                outdata[out.index(b, i, j)] = d;
            }
        }
    }

    return out;
}

struct HubertNoLayerNormConvLayer : LayerBase {
    int64_t layerId = 0;
    localnn::Tensor weight;

    HubertNoLayerNormConvLayer(int64_t layerId_) : layerId(layerId_) { }

    void prepare(string key) override {
        weight = loadData(key, "conv.weight");
    }
    
    localnn::Tensor forward(const localnn::Tensor &x) override {

        int64_t inSize = 1;
        if (layerId > 0) inSize = convDimensions[layerId-1];
        int64_t outSize = convDimensions[layerId];

        auto tmp = localConv1d_(x, inSize, outSize,
                                convKernels[layerId], convStrides[layerId],
                                0, 1, weight, nullptr);
        localGelu_(tmp);
        
        return tmp;
    }
};

struct HubertGroupNormConvLayer : LayerBase {

    int64_t layerId = 0;
    localnn::Tensor convWeight;
    localnn::Tensor normWeight;
    localnn::Tensor normBias;
    
    HubertGroupNormConvLayer(int64_t layerId_) : layerId(layerId_) { }

    void prepare(string key) override {
        convWeight = loadData(key, "conv.weight");
        normWeight = loadData(key, "layer_norm.weight");
        normBias = loadData(key, "layer_norm.bias");
    }
    
    localnn::Tensor forward(const localnn::Tensor &x) override {

        int64_t inSize = 1;
        if (layerId > 0) inSize = convDimensions[layerId-1];
        int64_t outSize = convDimensions[layerId];

        auto tmp = localConv1d_(x, inSize, outSize,
                                convKernels[layerId], convStrides[layerId],
                                0, 1, convWeight, nullptr);

        localLayerNorm_(tmp, normWeight, normBias, true);
        localGelu_(tmp);
        
        return tmp;
    }
};

struct HubertFeatureEncoder : LayerBase {

    vector<shared_ptr<LayerBase>> layers;

    HubertFeatureEncoder() {
        layers.push_back(make_shared<HubertGroupNormConvLayer>(0));
        for (int i = 1; i < convDimensions.size(); ++i) {
            layers.push_back(make_shared<HubertNoLayerNormConvLayer>(i));
        }
    }

    void prepare(string key) override {
        for (int i = 0; i < layers.size(); ++i) {
            layers[i]->prepare(key + ".conv_layers." + to_string(i));
        }
    }

    localnn::Tensor forward(const localnn::Tensor &x) {
        localnn::Tensor t = x;
        for (int i = 0; i < layers.size(); ++i) {
            t = layers[i]->forward(t);
        }
        return t;
    }
};

struct MERTFeatureProjection : LayerBase {

    localnn::Tensor normWeight;
    localnn::Tensor normBias;
    localnn::Tensor linearWeight;
    localnn::Tensor linearBias;
    
    MERTFeatureProjection() { }

    void prepare(string key) override {
        normWeight = loadData(key, "layer_norm.weight");
        normBias = loadData(key, "layer_norm.bias");
        linearWeight = loadData(key, "projection.weight");
        linearBias = loadData(key, "projection.bias");
    }

    localnn::Tensor forward(const localnn::Tensor &x) {
        localnn::Tensor t = x;
        localLayerNorm_(t, normWeight, normBias, false);
        t = localLinear_(t, linearWeight, linearBias);
        return t;
    }
};

struct HubertSamePadLayer : LayerBase {

    HubertSamePadLayer() { }

    void prepare(string) override { }
    
    localnn::Tensor forward(const localnn::Tensor &t) {
        if (nConvPosEmbeddings % 2 != 0) {
            return t;
        }
        if (t.rank != 3) {
            cerr << "error: wrong rank" << endl;
            throw std::runtime_error("shape");
        }
        auto outsizes = t.sizes;
        --(*outsizes.rbegin());
        auto out = localnn::Tensor(outsizes);
        auto tdata = t.data();
        auto outdata = out.mutableData();
        for (int i = 0; i < t.sizes[0]; ++i) {
            for (int j = 0; j < t.sizes[1]; ++j) {
#pragma GCC ivdep
                for (int k = 0; k+1 < t.sizes[2]; ++k) {
                    outdata[out.index(i, j, k)] = tdata[t.index(i, j, k)];
                }
            }
        }
        return out;
    }
    
};

struct HubertPositionalConvEmbedding : LayerBase {

    HubertSamePadLayer padding;
    localnn::Tensor weight;
    localnn::Tensor bias;
    
    HubertPositionalConvEmbedding() { }

    void prepare(string key) override {
        // NB the PyTorch implementation uses a weight_norm
        // parameterisation - we don't have that here so when
        // exporting from PyTorch we must be sure to call
        // parametrize.remove_parametrizations on this layer first
        weight = loadData(key, "conv.weight");
        bias = loadData(key, "conv.bias");
        padding.prepare(key + ".padding");
    }

    localnn::Tensor forward(const localnn::Tensor &in) {
        auto hidden_states = localTranspose12of3(in);
        hidden_states = localConv1d_
            (hidden_states, hiddenSize, hiddenSize,
             nConvPosEmbeddings, 1, nConvPosEmbeddings/2,
             nConvPosEmbeddingGroups,
             weight, &bias);

        hidden_states = padding.forward(hidden_states);
        localGelu_(hidden_states);
        hidden_states = localTranspose12of3(hidden_states);
        return hidden_states;
    }
};

struct HubertAttention : LayerBase {

    localnn::Tensor kWeight;
    localnn::Tensor kBias;
    localnn::Tensor vWeight;
    localnn::Tensor vBias;
    localnn::Tensor qWeight;
    localnn::Tensor qBias;
    localnn::Tensor outWeight;
    localnn::Tensor outBias;
    
    int64_t embed_dim;
    int64_t num_heads;
    int64_t head_dim;
    double scaling;
    
    HubertAttention() :
        embed_dim(hiddenSize),
        num_heads(nAttentionHeads),
        head_dim(embed_dim / num_heads),
        scaling(pow(head_dim, -0.5)) {
    }

    void prepare(string key) override {
        kWeight = loadData(key, "k_proj.weight");
        kBias = loadData(key, "k_proj.bias");
        vWeight = loadData(key, "v_proj.weight");
        vBias = loadData(key, "v_proj.bias");
        qWeight = loadData(key, "q_proj.weight");
        qBias = loadData(key, "q_proj.bias");
        outWeight = loadData(key, "out_proj.weight");
        outBias = loadData(key, "out_proj.bias");
    }

    localnn::Tensor shape(const localnn::Tensor &x, int64_t seq_len, int64_t bsz) {
        vector<int64_t> dim { bsz, seq_len, num_heads, head_dim };
        return localTranspose12of4(localReshape(x, dim));
    }
    
    localnn::Tensor forward(const localnn::Tensor &hidden_states) {
        
        auto bsz = hidden_states.sizes[0];
        auto tgt_len = hidden_states.sizes[1];

        auto query_states = localLinear_(hidden_states, qWeight, qBias);

        //!!!
        {
            auto qdata = query_states.mutableData();
            for (int64_t i = 0; i < query_states.numel(); ++i) {
                qdata[i] *= scaling;
            }
        }

        auto key_states =
            shape(localLinear_(hidden_states, kWeight, kBias),
                  -1, bsz);

        auto value_states =
            shape(localLinear_(hidden_states, vWeight, vBias),
                  -1, bsz);

        vector<int64_t> proj_shape { bsz * num_heads, -1, head_dim };
        query_states = localReshape(shape(query_states, tgt_len, bsz), proj_shape);
        
        key_states = localReshape(key_states, proj_shape);
        value_states = localReshape(value_states, proj_shape);
        
        int64_t src_len = key_states.sizes[1];
        auto attn_weights = localBMM_(query_states, localTranspose12of3(key_states));

        vector<int64_t> expected { bsz * num_heads, tgt_len, src_len };
        if (attn_weights.sizes != expected) {
            throw std::runtime_error("shape");
        }

        localSoftmax_(attn_weights);

        auto attn_output = localBMM_(attn_weights, value_states);
        
        expected = { bsz * num_heads, tgt_len, head_dim };
        if (attn_output.sizes != expected) {
            throw std::runtime_error("shape");
        }

        attn_output = localReshape(attn_output, { bsz, num_heads, tgt_len, head_dim });

        attn_output = localTranspose12of4(attn_output);
        attn_output = localReshape(attn_output, { bsz, tgt_len, embed_dim });
        attn_output = localLinear_(attn_output, outWeight, outBias);

        return attn_output;
    }
    
};

struct HubertFeedForward : LayerBase {

    localnn::Tensor intermediateWeight;
    localnn::Tensor intermediateBias;
    localnn::Tensor outputWeight;
    localnn::Tensor outputBias;

    HubertFeedForward() { }

    void prepare(string key) override {
        intermediateWeight = loadData(key, "intermediate_dense.weight");
        intermediateBias = loadData(key, "intermediate_dense.bias");
        outputWeight = loadData(key, "output_dense.weight");
        outputBias = loadData(key, "output_dense.bias");
    }

    localnn::Tensor forward(const localnn::Tensor &in) {

        auto hidden_states = localLinear_
            (in, intermediateWeight, intermediateBias);

        localGelu_(hidden_states);
        
        hidden_states = localLinear_
            (hidden_states, outputWeight, outputBias);
        
        return hidden_states;
    }
};


struct HubertEncoderLayer : LayerBase {

    HubertAttention attention;
    HubertFeedForward feed_forward;

    localnn::Tensor normWeight;
    localnn::Tensor normBias;
    localnn::Tensor finalNormWeight;
    localnn::Tensor finalNormBias;

    HubertEncoderLayer() { }

    void prepare(string key) override {
        normWeight = loadData(key, "layer_norm.weight");
        normBias = loadData(key, "layer_norm.bias");
        finalNormWeight = loadData(key, "final_layer_norm.weight");
        finalNormBias = loadData(key, "final_layer_norm.bias");
        attention.prepare(key + ".attention");
        feed_forward.prepare(key + ".feed_forward");
    }

    localnn::Tensor forward(const localnn::Tensor &in) {
        
        localnn::Tensor attn_residual = in;

        localnn::Tensor hidden_states = attention.forward(in);

        //!!! oh come on, let's have a function
        {
            auto rdata = attn_residual.data();
            auto hdata = hidden_states.mutableData();
            for (int64_t i = 0; i < hidden_states.numel(); ++i) {
                hdata[i] += rdata[i];
            }
        }

        localLayerNorm_(hidden_states, normWeight, normBias, false);

        localnn::Tensor ff = feed_forward.forward(hidden_states);

        {
            auto fdata = ff.data();
            auto hdata = hidden_states.mutableData();
            for (int64_t i = 0; i < hidden_states.numel(); ++i) {
                hdata[i] += fdata[i];
            }
        }

        localLayerNorm_(hidden_states, finalNormWeight, finalNormBias, false);
        
        return hidden_states;
    }
    
};

struct HubertEncoder : Module {

    HubertPositionalConvEmbedding pos_conv_embed;
    vector<shared_ptr<HubertEncoderLayer>> layers;

    localnn::Tensor normWeight;
    localnn::Tensor normBias;

    HubertEncoder() {
        for (int i = 0; i < nHiddenLayers; ++i) {
            layers.push_back(make_shared<HubertEncoderLayer>());
        }
    }

    void prepare(string key) override {
        normWeight = loadData(key, "layer_norm.weight");
        normBias = loadData(key, "layer_norm.bias");
        pos_conv_embed.prepare(key + ".pos_conv_embed");
        for (int i = 0; i < nHiddenLayers; ++i) {
            layers[i]->prepare(key + ".layers." + to_string(i));
        }
    }
    
    vector<localnn::Tensor> forward(const localnn::Tensor &in) {

        auto hidden_states = pos_conv_embed.forward(in);

        {
            auto idata = in.data();
            auto hdata = hidden_states.mutableData();
            for (int64_t i = 0; i < hidden_states.numel(); ++i) {
                hdata[i] += idata[i];
            }
        }

        localLayerNorm_(hidden_states, normWeight, normBias, false);
        
        vector<localnn::Tensor> all_hidden_states;
        all_hidden_states.push_back(hidden_states);

        for (int i = 0; i < layers.size(); ++i) {
            hidden_states = layers[i]->forward(hidden_states);
            all_hidden_states.push_back(hidden_states);
        }

        return all_hidden_states;
    }
};

struct MERT : Module {
    
    HubertFeatureEncoder featureExtractor;
    MERTFeatureProjection featureProjection;
    HubertEncoder encoder;

    MERT() { }

    void prepare(string key) override {
        featureExtractor.prepare(key + ".feature_extractor");
        featureProjection.prepare(key + ".feature_projection");
        encoder.prepare(key + ".encoder");
    }

    vector<localnn::Tensor> forward(const localnn::Tensor &input_values) {
        auto features = featureExtractor.forward(input_values);
        features = localTranspose12of3(features);
        auto hidden_states = featureProjection.forward(features);
        auto encoder_outputs = encoder.forward(hidden_states);
        return encoder_outputs;
    }
};

int main(int argc, char **argv)
{
    MERT mert;

    mert.prepare("");

//    string testfile = "stairway-intro-16k-mono.wav";
    string testfile = "../data/gerudo.wav";
    
    SF_INFO sfinfo;
    SNDFILE *sf = sf_open(testfile.c_str(), SFM_READ, &sfinfo);
    if (!sf) {
        cerr << "Failed to open test file " << testfile << endl;
        return 2;
    }
    if (sfinfo.frames == 0) {
        cerr << "No frame count in test file " << testfile << endl;
        return 2;
    }
    vector<float> data(sfinfo.frames);
    if (auto count = sf_readf_float(sf, data.data(), sfinfo.frames) != sfinfo.frames) {
        cerr << "Failed to read whole test file " << testfile
             << ": read " << count << " of " << sfinfo.frames << " frames "
             << endl;
        return 2;
    }
    sf_close(sf);

    cerr << "read " << data.size() << "-sample vector from test file "
         << testfile << endl;

    localnn::Tensor input({ 1, 1, data.size() }, data);
    
    vector<localnn::Tensor> output = mert.forward(input);

    cerr << "received " << output.size() << " tensors as output" << endl;

    dump(output[0], "experiment-out-0");
    dump(output[12], "experiment-out-12");
}

