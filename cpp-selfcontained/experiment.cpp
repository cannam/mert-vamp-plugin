
#include <torch/nn.h>
#include <torch/serialize.h>

#include <iostream>
#include <fstream>

#include <sndfile.h>

#include "withoutlib.cpp"

#include "../data/weights.hpp"

using namespace std;
using namespace torch;

// For inference only - I've omitted some logic that I believe is only
// used in training (dropout, certain types of masking)

static const int64_t vocabSize { 32 };
static const int64_t hiddenSize { 768 };
static const int64_t nHiddenLayers { 12 };
static const int64_t nAttentionHeads { 12 };
static const int64_t nConvPosEmbeddings { 128 };
static const int64_t nConvPosEmbeddingGroups { 16 };
static const int64_t intermediateSize { 3072 };
static const vector<int64_t> convDimensions { 512, 512, 512, 512, 512, 512, 512 };
static const vector<int64_t> convStrides { 5, 2, 2, 2, 2, 2, 2 };
static const vector<int64_t> convKernels { 10, 3, 3, 3, 3, 2, 2 };

#define DEBUG_TENSOR_SHAPES 1
#ifdef DEBUG_TENSOR_SHAPES
#include <cxxabi.h>
#endif

localnn::Tensor localFromTorch(Tensor t)
{
    t = t.to(kCPU);
    return localnn::tensorFromData
        (t.sizes().size(),
         t.sizes().data(),
         t.strides().data(),
         t.data_ptr<float>());
}

Tensor torchFromLocal(localnn::Tensor t)
{
    return torch::from_blob(t.data.data(), { t.sizes.data(), t.sizes.size() }).clone();
}

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
    
/*
    int nrows = t.sizes()[1];
    int ncols = t.sizes()[2];
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
            csv << v[i * ncols + j];
            if (j + 1 < ncols) {
                csv << ",";
            } else {
                csv << endl;
            }
        }
    }
*/
}

void dump(Tensor t, string filebase)
{
    // We assume t has channels as first dimension but only has one
    // channel, so we can treat as 2-d
    
    // We did nothing to move it away from the CPU, but just in case
    // that changes later!
    t = t.to(kCPU).contiguous();

    cerr << "will dump tensor of sizes " << t.sizes() << endl;
    localnn::Tensor tt = localFromTorch(t);

    dump(tt, filebase);
}

void localLinearImpl(const localnn::Tensor &in, int64_t inbase,
                     localnn::Tensor &out, int64_t outbase,
                     const localnn::Tensor &weight,
                     const localnn::Tensor &bias,
                     int64_t rank,
                     int64_t rix)
{
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
                x += weight.at(j, i) * in.data[inbase + i * instride];
            }
            x += bias.at(j);
            out.data[outbase + j * outstride] = x;
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
    auto out = localnn::Tensor::empty(outsizes);
    int rix = 0;
    while (in.sizes[rix] == 1) ++rix;
    localLinearImpl(in, 0,
                    out, 0,
                    weight, bias,
                    rank, rix);
    return out;
}

Tensor localLinear(Tensor x, Tensor weight, Tensor bias)
{
    auto tx = localFromTorch(x);
    auto tw = localFromTorch(weight);
    auto tb = localFromTorch(bias);
    auto result = localLinear_(tx, tw, tb);
    return torchFromLocal(result);
}

struct LayerBase : nn::Module {
    virtual localnn::Tensor forwardImpl(const localnn::Tensor &x) = 0;

    virtual Tensor forward(Tensor x) {
#ifdef DEBUG_TENSOR_SHAPES
        auto inshape = x.sizes();
        auto instrides = x.strides();
#endif

        localnn::Tensor in = localFromTorch(x.contiguous());
        localnn::Tensor out = forwardImpl(in);
        x = torchFromLocal(out);
        
#ifdef DEBUG_TENSOR_SHAPES
        cerr << abi::__cxa_demangle(typeid(*this).name(), nullptr, nullptr, nullptr) << ": " << inshape << " (" << instrides << ") -> " << x.sizes() << " (" << x.strides() << ")" << endl;
#endif

        return x;
    }
};

void localGelu_(localnn::Tensor &t)
{
    const double alpha = M_SQRT1_2;

    //!!! +omp
    
    for (int64_t i = 0; i < t.data.size(); ++i) {

        double x = t.data[i];
        x = x * 0.5 * (1.0 + std::erf(x * alpha));
        t.data[i] = float(x);
    }
}
    
void localGelu(Tensor &tt)
{
    //!!! the fact that this .contiguous is necessary suggests we've
    //!!! screwed up localFromTorch somehow
    localnn::Tensor t = localFromTorch(tt.contiguous());
    localGelu_(t);
    tt = torchFromLocal(t);
}

// We always copy for reshape and transpose, because we can only
// handle contiguous layouts in some other functions

localnn::Tensor localReshape(const localnn::Tensor &t, vector<int64_t> outsizes)
{
    cerr << "localReshape: proposed sizes: " << outsizes << ", numel: " << t.numel() << endl;
    //!!! clumsy but test this first
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
    cerr << "localReshape: adjusted sizes: " << outsizes << endl;
    localnn::Tensor out = localnn::Tensor::empty(outsizes);
    int64_t n = t.numel();
    if (out.numel() != n) {
        cerr << "wrong total number of elements in reshape (" << out.numel() << " vs " << n << ")" << endl;
        throw std::runtime_error("shape");
    }
    out.data = t.data;
    return out;
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
    localnn::Tensor out = localnn::Tensor::empty(outsizes);
    for (int64_t i = 0; i < a; ++i) {
        for (int64_t j = 0; j < b; ++j) {
#pragma GCC ivdep
            for (int64_t k = 0; k < c; ++k) {
                out.data[out.index(i, k, j)] = t.data[t.index(i, j, k)];
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
    localnn::Tensor out = localnn::Tensor::empty(outsizes);
    for (int64_t i = 0; i < a; ++i) {
        for (int64_t j = 0; j < b; ++j) {
            for (int64_t k = 0; k < c; ++k) {
#pragma GCC ivdep
                for (int64_t m = 0; m < d; ++m) {
                    out.data[out.index(i, k, j, m)] =
                        t.data[t.index(i, j, k, m)];
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

    localnn::Tensor out = localnn::Tensor::empty({ t.sizes[0], ch_out, l_out });
    
    const float *bbase = nullptr;
    if (bp) {
        bbase = bp->data.data();
    }

    for (int64_t b = 0; b < t.sizes[0]; ++b) {
        for (int64_t g = 0; g < groups; ++g) {
            int c0 = g * (ch_out / groups);
            int k0 = g * (ch_in / groups);
            int g0 = g * ((out.numel() / t.sizes[0]) / groups);
#pragma omp parallel for
            for (int64_t c = 0; c < ch_out / groups; ++c) {

                float *const outbase =
                    out.data.data() + g0 + out.index(b, c);

                for (int64_t k = 0; k < ch_in / groups; ++k) {
                    const float *const wbase =
                        w.data.data() + w.index(c + c0, k);
                    const float *const tbase =
                        t.data.data() + t.index(b, k + k0);
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

Tensor localConv1d(Tensor tt, int64_t ch_in, int64_t ch_out, int64_t ksize,
                   int64_t stride, int64_t padding, int64_t groups,
                   Tensor weightt, Tensor *biasp)
{
    cerr << "in shape = " << tt.sizes() << endl;
    cerr << "weight shape = " << weightt.sizes() << endl;
    cerr << "groups = " << groups << endl;
    
    localnn::Tensor t = localFromTorch(tt.contiguous());
    localnn::Tensor w = localFromTorch(weightt);

    localnn::Tensor bias;
    if (biasp) {
        bias = localFromTorch(*biasp);
    }

    auto out = localConv1d_(t, ch_in, ch_out, ksize, stride, padding, groups,
                            w, biasp ? &bias : nullptr);
            
    Tensor result = torchFromLocal(out);
    cerr << "out shape = " << result.sizes() << endl;
    return result;
}

void localLayerNorm_(localnn::Tensor &t,
                     const localnn::Tensor &weight, const localnn::Tensor &bias,
                     bool weightsPerInstance)
{
    int64_t h = *t.sizes.rbegin();
    int64_t m = t.numel() / h;

    for (int64_t j = 0; j < m; ++j) {

        float *base = t.data.data() + j * h;
        
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

void localLayerNorm(Tensor &tt, Tensor weightt, Tensor biast,
                    bool weightsPerInstance)
{
    // Fixed to last dimension
    localnn::Tensor t = localFromTorch(tt.contiguous()); //!!!???
    localnn::Tensor weight = localFromTorch(weightt);
    localnn::Tensor bias = localFromTorch(biast);

    localLayerNorm_(t, weight, bias, weightsPerInstance);
    
    tt = torchFromLocal(t);
}

localnn::Tensor localBMM_(const localnn::Tensor &t, const localnn::Tensor &m)
{
    if (t.rank != 3 || m.rank != 3) {
        cerr << "unsupported rank (" << t.rank << " or "
             << m.rank << ", should both be 3)" << endl;
        throw std::runtime_error("shape");
    }        
    
    localnn::Tensor out = localnn::Tensor::empty
        ({ t.sizes[0], t.sizes[1], m.sizes[2] });

    if (t.sizes[2] != m.sizes[1]) {
        cerr << "incompatible sizes (" << t.sizes[2] << " != "
             << m.sizes[1] << ")" << endl;
        throw std::runtime_error("shape");
    }
    
    for (int64_t b = 0; b < t.sizes[0]; ++b) {
#pragma omp parallel for
        for (int64_t i = 0; i < t.sizes[1]; ++i) {
            for (int64_t j = 0; j < m.sizes[2]; ++j) {
                double d = 0.0;
#pragma GCC ivdep
                for (int64_t k = 0; k < m.sizes[1]; ++k) {
                    d += t.at(b, i, k) * m.at(b, k, j);
                }
                out.data[out.index(b, i, j)] = d;
            }
        }
    }

    return out;
}

Tensor localBMM(Tensor tt, Tensor mt)
{
    localnn::Tensor t = localFromTorch(tt.contiguous()); //!!!???
    localnn::Tensor m = localFromTorch(mt.contiguous());

    auto out = localBMM_(t, m);
    
    return torchFromLocal(out);
}

struct HubertNoLayerNormConvLayerImpl : LayerBase {
    int64_t layerId = 0;
    nn::Conv1d conv = nullptr;

    HubertNoLayerNormConvLayerImpl(int64_t layerId_) {
        layerId = layerId_;
        int64_t inSize = 1;
        if (layerId > 0) inSize = convDimensions[layerId-1];
        int64_t outSize = convDimensions[layerId];

        nn::Conv1dOptions options =
            nn::Conv1dOptions(inSize, outSize, convKernels[layerId])
            .stride(convStrides[layerId]).bias(false);
        conv = register_module("conv", nn::Conv1d(options));
    }
    
    localnn::Tensor forwardImpl(const localnn::Tensor &x) override {

        int64_t inSize = 1;
        if (layerId > 0) inSize = convDimensions[layerId-1];
        int64_t outSize = convDimensions[layerId];

        localnn::Tensor w = localFromTorch(conv->weight);
        
        auto tmp = localConv1d_(x, inSize, outSize,
                                convKernels[layerId], convStrides[layerId],
                                0, 1, w, nullptr);
        localGelu_(tmp);
        
        return tmp;
    }
};

TORCH_MODULE(HubertNoLayerNormConvLayer);

struct HubertGroupNormConvLayerImpl : LayerBase {

    int64_t layerId = 0;
    nn::Conv1d conv = nullptr;
    nn::GroupNorm layer_norm = nullptr;
    
    HubertGroupNormConvLayerImpl(int64_t layerId_) {
        layerId = layerId_;
        int64_t inSize = 1;
        if (layerId > 0) inSize = convDimensions[layerId-1];
        int64_t outSize = convDimensions[layerId];

        nn::Conv1dOptions options =
            nn::Conv1dOptions(inSize, outSize, convKernels[layerId])
            .stride(convStrides[layerId]).bias(false);
        conv = register_module("conv", nn::Conv1d(options));

        nn::GroupNormOptions normOptions =
            nn::GroupNormOptions(outSize, outSize).affine(true);
        layer_norm = register_module("layer_norm", nn::GroupNorm(normOptions));
    }
    
    localnn::Tensor forwardImpl(const localnn::Tensor &x) override {

        int64_t inSize = 1;
        if (layerId > 0) inSize = convDimensions[layerId-1];
        int64_t outSize = convDimensions[layerId];

        localnn::Tensor w = localFromTorch(conv->weight);

        auto tmp = localConv1d_(x, inSize, outSize,
                                convKernels[layerId], convStrides[layerId],
                                0, 1, w, nullptr);

        w = localFromTorch(layer_norm->weight);
        auto b = localFromTorch(layer_norm->bias);
        
        localLayerNorm_(tmp, w, b, true);
        localGelu_(tmp);
        
        return tmp;
    }
};

TORCH_MODULE(HubertGroupNormConvLayer);

struct HubertFeatureEncoderImpl : LayerBase {

    nn::ModuleList layers;

    HubertFeatureEncoderImpl() {
        layers = register_module("conv_layers", nn::ModuleList());
        layers->push_back(HubertGroupNormConvLayer(0));
        for (int i = 1; i < convDimensions.size(); ++i) {
            layers->push_back(HubertNoLayerNormConvLayer(i));
        }
    }

    localnn::Tensor forwardImpl(const localnn::Tensor &x) {
        localnn::Tensor t = x;
        for (int i = 0; i < layers->size(); ++i) {
            if (auto layer = layers[i]->as<LayerBase>()) {
                t = layer->forwardImpl(t);
            } else {
                cerr << "HubertFeatureEncoder: Unexpected type for layer " << i
                     << endl;
                throw std::runtime_error("wrong type");
            }
        }
        return t;
    }
};

TORCH_MODULE(HubertFeatureEncoder);

struct MERTFeatureProjectionImpl : LayerBase {

    nn::LayerNorm layer_norm = nullptr;
    nn::Linear linear = nullptr;

    //!!! CQT would go here too but it doesn't seem to be in the
    //!!! config for the reference weights I have?
    
    MERTFeatureProjectionImpl() {
        int64_t inSize = *convDimensions.rbegin();
        int64_t outSize = hiddenSize;
        nn::LayerNormOptions options({ inSize });
        layer_norm = register_module("layer_norm", nn::LayerNorm(options));
        linear = register_module("projection", nn::Linear(inSize, outSize));
    }

    localnn::Tensor forwardImpl(const localnn::Tensor &x) {
        localnn::Tensor t = x;
        localLayerNorm_(t,
                        localFromTorch(layer_norm->weight),
                        localFromTorch(layer_norm->bias),
                        false);
        t = localLinear_(t,
                         localFromTorch(linear->weight),
                         localFromTorch(linear->bias));
        return t;
    }
        
};

TORCH_MODULE(MERTFeatureProjection);

struct HubertSamePadLayerImpl : LayerBase {

    HubertSamePadLayerImpl() { }

    localnn::Tensor forwardImpl(const localnn::Tensor &t) {
        if (nConvPosEmbeddings % 2 != 0) {
            return t;
        }
        if (t.rank != 3) {
            cerr << "error: wrong rank" << endl;
            throw std::runtime_error("shape");
        }
        auto outsizes = t.sizes;
        --(*outsizes.rbegin());
        auto out = localnn::Tensor::empty(outsizes);
        for (int i = 0; i < t.sizes[0]; ++i) {
            for (int j = 0; j < t.sizes[1]; ++j) {
#pragma GCC ivdep
                for (int k = 0; k+1 < t.sizes[2]; ++k) {
                    out.data[out.index(i, j, k)] = t.data[t.index(i, j, k)];
                }
            }
        }
        return out;
    }
    
};

TORCH_MODULE(HubertSamePadLayer);

struct HubertPositionalConvEmbeddingImpl : LayerBase {

    nn::Conv1d conv = nullptr;
    HubertSamePadLayer padding = nullptr;
    
    HubertPositionalConvEmbeddingImpl() {

        // The PyTorch implementation uses a weight_norm
        // parameterisation - we don't have that here so when
        // exporting from PyTorch we must be sure to call
        // parametrize.remove_parametrizations on this layer first
        
        nn::Conv1dOptions options =
            nn::Conv1dOptions(hiddenSize, hiddenSize, nConvPosEmbeddings)
            .padding(nConvPosEmbeddings/2)
            .groups(nConvPosEmbeddingGroups);
        conv = register_module("conv", nn::Conv1d(options));
        padding = register_module("padding", HubertSamePadLayer());
    }

    localnn::Tensor forwardImpl(const localnn::Tensor &in) {

        //!!!
        auto hidden_states = localFromTorch(torchFromLocal(in).transpose(1, 2).contiguous());

        auto w = localFromTorch(conv->weight);
        auto b = localFromTorch(conv->bias);
        
        hidden_states = localConv1d_(hidden_states, hiddenSize, hiddenSize,
                                     nConvPosEmbeddings, 1, nConvPosEmbeddings/2,
                                     nConvPosEmbeddingGroups,
                                     w, &b);

        hidden_states = padding->forwardImpl(hidden_states);

        localGelu_(hidden_states);

        //!!!
        hidden_states = localFromTorch(torchFromLocal(hidden_states).transpose(1, 2).contiguous());
        
        return hidden_states;
    }
    
};

TORCH_MODULE(HubertPositionalConvEmbedding);

struct HubertAttentionImpl : LayerBase {

    nn::Linear k_proj = nullptr;
    nn::Linear v_proj = nullptr;
    nn::Linear q_proj = nullptr;
    nn::Linear out_proj = nullptr;

    int64_t embed_dim;
    int64_t num_heads;
    int64_t head_dim;
    double scaling;
    
    HubertAttentionImpl() :
        embed_dim(hiddenSize),
        num_heads(nAttentionHeads),
        head_dim(embed_dim / num_heads),
        scaling(pow(head_dim, -0.5)) {
        k_proj = register_module("k_proj", nn::Linear(embed_dim, embed_dim));
        v_proj = register_module("v_proj", nn::Linear(embed_dim, embed_dim));
        q_proj = register_module("q_proj", nn::Linear(embed_dim, embed_dim));
        out_proj = register_module("out_proj", nn::Linear(embed_dim, embed_dim));
    }

    localnn::Tensor shape(const localnn::Tensor &x, int64_t seq_len, int64_t bsz) {
        vector<int64_t> dim { bsz, seq_len, num_heads, head_dim };
        return localTranspose12of4(localReshape(x, dim));
    }
    
    localnn::Tensor forwardImpl(const localnn::Tensor &hidden_states) {
        
        // "Input shape: Batch x Time x Channel"
        auto bsz = hidden_states.sizes[0];
        auto tgt_len = hidden_states.sizes[1];

        cerr << "HubertAttentionImpl::forwardImpl: input shape = "
             << hidden_states.sizes << endl;

        // input = [1, 1030, 768]   (where 1030 is the sequence length)
        
        //!!! why do we just reshape it twice? what is the point?
        auto query_states =
            localLinear_(hidden_states,
                         localFromTorch(q_proj->weight),
                         localFromTorch(q_proj->bias));
        
        for (int64_t i = 0; i < query_states.numel(); ++i) {
            query_states.data[i] *= scaling;
        }

        auto key_states =
            shape(localLinear_(hidden_states,
                               localFromTorch(k_proj->weight),
                               localFromTorch(k_proj->bias)),
                  -1, bsz);

//        auto key_states = shape(k_proj->forwardImpl(hidden_states), tgt_len, bsz);
        auto value_states =
            shape(localLinear_(hidden_states,
                               localFromTorch(v_proj->weight),
                               localFromTorch(v_proj->bias)),
                  -1, bsz);

//        auto value_states = shape(v_proj->forwardImpl(hidden_states), tgt_len, bsz);

        cerr << "q = " << query_states.sizes << ", k = " << key_states.sizes
             << ", v = " << value_states.sizes << endl;
        
        // q = [1, 1030, 768]
        // k = [1, 12, 1030, 64]
        // v = [1, 12, 1030, 64]
        
        vector<int64_t> proj_shape { bsz * num_heads, -1, head_dim };
        query_states = localReshape(shape(query_states, tgt_len, bsz), proj_shape);
        
        key_states = localReshape(key_states, proj_shape);
        value_states = localReshape(value_states, proj_shape);

        cerr << "q' = " << query_states.sizes << ", k' = " << key_states.sizes
             << ", v' = " << value_states.sizes << endl;

        // q' = [12, 1030, 64]
        // k' = [12, 1030, 64]
        // v' = [12, 1030, 64]
        
        int64_t src_len = key_states.sizes[1];
        auto attn_weights = localBMM_(query_states, localTranspose12of3(key_states));

        vector<int64_t> expected { bsz * num_heads, tgt_len, src_len };
        if (attn_weights.sizes != expected) {
            throw std::runtime_error("shape");
//            cerr << "Attention weights should be of size " << expected
//                 << " but are of size " << attn_weights.sizes() << endl;
        }

        // All masking etc omitted here (relevant only in training)

        //!!!
        attn_weights = localFromTorch(softmax(torchFromLocal(attn_weights), -1).contiguous());

        auto attn_output = localBMM_(attn_weights, value_states);
        
        expected = { bsz * num_heads, tgt_len, head_dim };
        if (attn_output.sizes != expected) {
            throw std::runtime_error("shape");
//            cerr << "Attention output should be of size " << expected
//                 << " but are of size " << attn_weights.sizes() << endl;
        }

        cerr << "output pre-reshape = " << attn_output.sizes << endl;

        // output pre-reshape = [12, 1030, 64]
        
//        attn_output = attn_output.view({ bsz, num_heads, tgt_len, head_dim });
        attn_output = localReshape(attn_output, { bsz, num_heads, tgt_len, head_dim });

        cerr << "output after view(" << bsz << "," << num_heads << "," << tgt_len << "," << head_dim << ") = " << attn_output.sizes << endl;

        // output after view = [1, 12, 1030, 64]
        
        attn_output = localTranspose12of4(attn_output);

        cerr << "output after transpose(1, 2) = " << attn_output.sizes << endl;

        // output after transpose = [1, 1030, 12, 64]
        
        attn_output = localReshape(attn_output, { bsz, tgt_len, embed_dim });

        cerr << "output after reshape(" << bsz << "," << tgt_len << "," << embed_dim << ") = " << attn_output.sizes << endl;

        // output after reshape = [1, 1030, 768]
        
//        attn_output = out_proj->forwardImpl(attn_output);
        attn_output = localLinear_(attn_output,
                                   localFromTorch(out_proj->weight),
                                   localFromTorch(out_proj->bias));
        
        cerr << "output after projection = " << attn_output.sizes << endl;
        
        // output after projection = [1, 1030, 768]

        return attn_output;
//        return localFromTorch(attn_output.contiguous()); //!!!
    }
    
};

TORCH_MODULE(HubertAttention);

struct HubertFeedForwardImpl : LayerBase {

    nn::Linear intermediate_dense = nullptr;
    nn::Linear output_dense = nullptr;

    HubertFeedForwardImpl() {
        intermediate_dense = register_module
            ("intermediate_dense", nn::Linear(hiddenSize, intermediateSize));
        output_dense = register_module
            ("output_dense", nn::Linear(intermediateSize, hiddenSize));
    }

    localnn::Tensor forwardImpl(const localnn::Tensor &in) {

        auto hidden_states = localLinear_
            (in,
             localFromTorch(intermediate_dense->weight),
             localFromTorch(intermediate_dense->bias));

        localGelu_(hidden_states);
        
        hidden_states = localLinear_
            (hidden_states,
             localFromTorch(output_dense->weight),
             localFromTorch(output_dense->bias));
        
        return hidden_states;
    }
    
};

TORCH_MODULE(HubertFeedForward);


struct HubertEncoderLayerImpl : LayerBase {

    HubertAttention attention = nullptr;
    nn::LayerNorm layer_norm = nullptr;
    HubertFeedForward feed_forward = nullptr;
    nn::LayerNorm final_layer_norm = nullptr;

    HubertEncoderLayerImpl() {
        //!!! in the Python there is a choice of "eager" -> HubertAttention,
        // "sdpa" -> HubertSdpaAttention, "flash" -> HubertFlashAttention2.
        // The layer actually constructed in our example is "eager"
        attention = register_module("attention", HubertAttention());
        nn::LayerNormOptions options({ hiddenSize });
        layer_norm = register_module("layer_norm", nn::LayerNorm(options));
        feed_forward = register_module("feed_forward", HubertFeedForward());
        final_layer_norm = register_module("final_layer_norm", nn::LayerNorm(options));
    }

    localnn::Tensor forwardImpl(const localnn::Tensor &in) {
        
        localnn::Tensor attn_residual = in;

        localnn::Tensor hidden_states = attention->forwardImpl(in);

        for (int64_t i = 0; i < hidden_states.numel(); ++i) {
            hidden_states.data[i] += attn_residual.data[i];
        }

        localLayerNorm_(hidden_states,
                        localFromTorch(layer_norm->weight),
                        localFromTorch(layer_norm->bias),
                        false);

        localnn::Tensor ff = feed_forward->forwardImpl(hidden_states);

        for (int64_t i = 0; i < hidden_states.numel(); ++i) {
            hidden_states.data[i] += ff.data[i];
        }

        localLayerNorm_(hidden_states,
                        localFromTorch(final_layer_norm->weight),
                        localFromTorch(final_layer_norm->bias),
                        false);
        
        return hidden_states;
    }
    
};

TORCH_MODULE(HubertEncoderLayer);

struct HubertEncoderImpl : nn::Module {

    HubertPositionalConvEmbedding pos_conv_embed = nullptr;
    nn::LayerNorm layer_norm = nullptr;
    nn::ModuleList layers;

    HubertEncoderImpl() {
        pos_conv_embed = register_module
            ("pos_conv_embed", HubertPositionalConvEmbedding());
        nn::LayerNormOptions options({ hiddenSize });
        layer_norm = register_module("layer_norm", nn::LayerNorm(options));
        layers = register_module("layers", nn::ModuleList());
        for (int i = 0; i < nHiddenLayers; ++i) {
            layers->push_back(HubertEncoderLayer());
        }
    }

    vector<localnn::Tensor> forward(const localnn::Tensor &in) {

        auto hidden_states = pos_conv_embed->forwardImpl(in);

        for (int64_t i = 0; i < hidden_states.numel(); ++i) {
            hidden_states.data[i] += in.data[i];
        }

        localLayerNorm_(hidden_states,
                        localFromTorch(layer_norm->weight),
                        localFromTorch(layer_norm->bias),
                        false);
        
        vector<localnn::Tensor> all_hidden_states;
        all_hidden_states.push_back(hidden_states);

        for (int i = 0; i < layers->size(); ++i) {
            if (auto layer = layers[i]->as<HubertEncoderLayer>()) {
                hidden_states = layer->forwardImpl(hidden_states);
                all_hidden_states.push_back(hidden_states);
            }
        }

        return all_hidden_states;
    }
    
};

TORCH_MODULE(HubertEncoder);

struct MERTImpl : nn::Module {
    
    HubertFeatureEncoder feature_extractor = nullptr;
    MERTFeatureProjection feature_projection = nullptr;
    HubertEncoder encoder = nullptr;

    MERTImpl() {
        feature_extractor = register_module("feature_extractor", HubertFeatureEncoder());
        feature_projection = register_module("feature_projection", MERTFeatureProjection());
        encoder = register_module("encoder", HubertEncoder());
    }

    vector<localnn::Tensor> forward(const localnn::Tensor &input_values) {
        auto extract_features = feature_extractor->forwardImpl(input_values);
        extract_features = localTranspose12of3(extract_features);
        auto hidden_states = feature_projection->forwardImpl(extract_features);
        auto encoder_outputs = encoder->forward(hidden_states);
        return encoder_outputs;
    }
    
};

TORCH_MODULE(MERT);

int main(int argc, char **argv)
{
    MERT mert;
    mert->eval();

    cerr << "Model parameters are:" << endl;
    auto params = mert->named_parameters();

    for (const auto &i : params) {
        std::string key = i.key();
        cerr << key;
        vector<int64_t> sizes;
        const float *data = lookup_model_data(key, sizes);
        if (!data) {
            cerr << " [! failed to load]";
        } else {
            Tensor t = torch::from_blob((float *)data, sizes);
            params[key].set_data(t);
        }
        cerr << endl;
    }

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

    //!!!
    localnn::Tensor input = localnn::Tensor::empty({ 1, 1, data.size() });
    input.data = data;
    
//    Tensor input = torch::from_blob(data.data(), { 1, 1, data.size() }).clone();

    vector<localnn::Tensor> output = mert(input);

    cerr << "received " << output.size() << " tensors as output" << endl;

    dump(output[0], "experiment-out-0");
    dump(output[12], "experiment-out-12");
}

