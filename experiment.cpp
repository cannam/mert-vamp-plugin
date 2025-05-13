
#include <torch/nn.h>
#include <torch/serialize.h>

#include <iostream>
#include <fstream>

#include <sndfile.h>

#include "withoutlib.cpp"
#include "withoutlib2.cpp"

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

localnn2::Tensor local2FromTorch(Tensor t)
{
    t = t.to(kCPU);
    return localnn2::fromData
        (t.sizes().size(),
         t.sizes().data(),
         t.strides().data(),
         t.data_ptr<float>());
}

Tensor torchFromLocal(localnn::Tensor t)
{
    return torch::from_blob(t.data.data(), { t.sizes.data(), t.sizes.size() }).clone();
}

Tensor torchFromLocal2(localnn2::Tensor t)
{
    vector<int64_t> sizes;
    auto data = localnn2::toData(t, sizes);
    Tensor tt = torch::from_blob(data.data(), { sizes.data(), sizes.size() }).clone();
    localnn2::Tensor check = local2FromTorch(tt);
    if (check != t) {
        throw std::runtime_error("check failed");
    }
    return tt;
}

void dump(Tensor t, string filebase)
{
    // We assume t has channels as first dimension but only has one
    // channel, so we can treat as 2-d
    
    // We did nothing to move it away from the CPU, but just in case
    // that changes later!
    t = t.to(kCPU).contiguous();

    vector<float> v(t.data_ptr<float>(), t.data_ptr<float>() + t.numel());
    
    string filename = filebase + ".csv";
    ofstream csv(filename);

    cerr << "will dump tensor of sizes " << t.sizes() << endl;

    localnn2::Tensor tt = local2FromTorch(t);
    localnn2::t_2 t2;
    switch (localnn2::rank(tt)) {
    case 1:
        t2 = { std::get<localnn2::t_1>(tt) };
        break;
    case 2:
        t2 = std::get<localnn2::t_2>(tt);
        break;
    case 3:
        t2 = std::get<localnn2::t_3>(tt)[0];
        break;
    case 4:
        t2 = std::get<localnn2::t_4>(tt)[0][0];
        break;
    default:
        throw std::runtime_error("unsupported rank in dump");
    }

    int nrows = t2.size();
    int ncols = t2[0].size();

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
            csv << t2.at(i).at(j);
            if (j + 1 < ncols) {
                csv << ",";
            } else {
                csv << endl;
            }
        }
    }
        
     
/*!!!
    localnn::Tensor tt = localFromTorch(t);

    int nrows = tt.sizes[1];
    int ncols = tt.sizes[2];

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
            csv << tt.at(0, i, j);
            if (j + 1 < ncols) {
                csv << ",";
            } else {
                csv << endl;
            }
        }
    }
*/
    
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

void localLinearImpl(const float *in,
                     const localnn::Tensor::ivec &insizes,
                     const localnn::Tensor::ivec &instrides,
                     float *out,
                     const localnn::Tensor::ivec &outsizes,
                     const localnn::Tensor::ivec &outstrides,
                     const localnn::Tensor &weight,
                     const localnn::Tensor &bias,
                     int64_t rank,
                     int64_t rix)
{
    if (rix + 1 == rank) {
        for (int64_t i = 0; i < insizes[rix]; ++i) {
            for (int64_t j = 0; j < outsizes[rix]; ++j) {
                out[j * outstrides[rix]] += weight.at(j, i) * in[i * instrides[rix]];
            }
        }
        for (int64_t j = 0; j < outsizes[rix]; ++j) {
            out[j * outstrides[rix]] += bias.at(j);
        }
    } else {
        for (int rc = 0; rc < insizes[rix]; ++rc) {
            localLinearImpl(in + rc * instrides[rix],
                            insizes,
                            instrides,
                            out + rc * outstrides[rix],
                            outsizes,
                            outstrides,
                            weight, bias,
                            rank,
                            rix + 1);
        }
    }
}

localnn2::t_1 localLinear2Impl(const localnn2::t_1 &in,
                               const localnn2::t_2 &w,
                               const localnn2::t_1 &b)
{
    size_t in_size = in.size();
    size_t out_size = w.size();
    localnn2::t_1 out(out_size, 0.f);
    for (size_t i = 0; i < in_size; ++i) {
        for (size_t j = 0; j < out_size; ++j) {
            out[j] += w[j][i] * in[i];
        }
    }
    for (size_t j = 0; j < out_size; ++j) {
        out[j] += b[j];
    }
    return out;
}

localnn2::t_2 localLinear2Impl_2(const localnn2::t_2 &in,
                                 const localnn2::t_2 &w,
                                 const localnn2::t_1 &b)
{
    localnn2::t_2 out;
    for (auto t1 : in) {
        out.push_back(localLinear2Impl(t1, w, b));
    }
    return out;
}

localnn2::t_3 localLinear2Impl_3(const localnn2::t_3 &in,
                                 const localnn2::t_2 &w,
                                 const localnn2::t_1 &b)
{
    localnn2::t_3 out;
    for (auto t2 : in) {
        out.push_back(localLinear2Impl_2(t2, w, b));
    }
    return out;
}

localnn2::t_4 localLinear2Impl_4(const localnn2::t_4 &in,
                                 const localnn2::t_2 &w,
                                 const localnn2::t_1 &b)
{
    localnn2::t_4 out;
    for (auto t3 : in) {
        out.push_back(localLinear2Impl_3(t3, w, b));
    }
    return out;
}

Tensor localLinear2(Tensor x, Tensor weight, Tensor bias)
{
    auto tx = local2FromTorch(x);
    auto tw = std::get<localnn2::t_2>(local2FromTorch(weight));
    auto tb = std::get<localnn2::t_1>(local2FromTorch(bias));
    localnn2::Tensor result;
    switch (localnn2::rank(tx)) {
    case 1:
        result = localLinear2Impl(std::get<localnn2::t_1>(tx), tw, tb);
        break;
    case 2:
        result = localLinear2Impl_2(std::get<localnn2::t_2>(tx), tw, tb);
        break;
    case 3:
        result = localLinear2Impl_3(std::get<localnn2::t_3>(tx), tw, tb);
        break;
    case 4:
        result = localLinear2Impl_4(std::get<localnn2::t_4>(tx), tw, tb);
        break;
    default:
        throw std::runtime_error("unsupported rank in localLinear2");
    }
    return torchFromLocal2(result);
}

Tensor localLinear(Tensor x, Tensor weight, Tensor bias)
{
    auto tx = localFromTorch(x);
    auto tw = localFromTorch(weight);
    auto tb = localFromTorch(bias);
    auto rank = tx.rank;
    auto outsizes = tx.sizes;
    outsizes[rank-1] = tw.sizes[0];
    auto out = localnn::Tensor::empty(outsizes);
    cerr << "new empty tensor: " << out << endl;
    localLinearImpl(tx.data.data(),
                    tx.sizes,
                    tx.strides,
                    out.data.data(),
                    out.sizes,
                    out.strides,
                    tw, tb,
                    rank, 0);
    auto result = torchFromLocal(out);
    dump(result, "tmp");
    return result;
}

struct LayerBase : nn::Module {
    virtual Tensor forwardImpl(Tensor x) = 0;
    
    virtual Tensor forward(Tensor x) {
#ifdef DEBUG_TENSOR_SHAPES
        auto inshape = x.sizes();
        auto instrides = x.strides();
#endif
        x = forwardImpl(x);
#ifdef DEBUG_TENSOR_SHAPES
        cerr << abi::__cxa_demangle(typeid(*this).name(), nullptr, nullptr, nullptr) << ": " << inshape << " (" << instrides << ") -> " << x.sizes() << " (" << x.strides() << ")" << endl;
#endif

        return x;
    }
};

struct HubertNoLayerNormConvLayerImpl : LayerBase {
    nn::Conv1d conv = nullptr;

    HubertNoLayerNormConvLayerImpl(int64_t layerId) {
        int64_t inSize = 1;
        if (layerId > 0) inSize = convDimensions[layerId-1];
        int64_t outSize = convDimensions[layerId];

        nn::Conv1dOptions options =
            nn::Conv1dOptions(inSize, outSize, convKernels[layerId])
            .stride(convStrides[layerId]).bias(false);
        conv = register_module("conv", nn::Conv1d(options));
    }
    
    Tensor forwardImpl(Tensor x) override {
        x = conv(x);
        x = gelu(x);
        return x;
    }
};

TORCH_MODULE(HubertNoLayerNormConvLayer);

struct HubertGroupNormConvLayerImpl : LayerBase {
    nn::Conv1d conv = nullptr;
    nn::GroupNorm layer_norm = nullptr;
    
    HubertGroupNormConvLayerImpl(int64_t layerId) {
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
    
    Tensor forwardImpl(Tensor x) override {
        x = conv(x);
        x = layer_norm(x);
        x = gelu(x);
        return x;
    }
};

TORCH_MODULE(HubertGroupNormConvLayer);

struct HubertFeatureEncoderImpl : LayerBase {

    nn::ModuleList layers;

    HubertFeatureEncoderImpl() {
        layers = register_module("conv_layers", nn::ModuleList());
        layers->push_back(register_module("0", HubertGroupNormConvLayer(0)));
        for (int i = 1; i < convDimensions.size(); ++i) {
            layers->push_back(register_module(to_string(i), HubertNoLayerNormConvLayer(i)));
        }
    }

    Tensor forwardImpl(Tensor x) {
        for (int i = 0; i < layers->size(); ++i) {
            if (auto layer = layers[i]->as<LayerBase>()) {
                x = layer->forward(x);
            } else {
                cerr << "HubertFeatureEncoder: Unexpected type for layer " << i
                     << endl;
            }
        }
        return x;
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

    Tensor forwardImpl(Tensor x) {
        x = layer_norm(x);
//        x = linear(x);
        x = localLinear2(x, linear->weight, linear->bias);
        return x;
    }
        
};

TORCH_MODULE(MERTFeatureProjection);

struct HubertSamePadLayerImpl : LayerBase {

    HubertSamePadLayerImpl() { }

    Tensor forwardImpl(Tensor x) {
        if (nConvPosEmbeddings % 2 == 0) {
            // [:, :, : -1]
            return x.index({
                    indexing::Slice(),
                    indexing::Slice(),
                    indexing::Slice(indexing::None, -1)
                });
        } else {
            return x;
        }
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

    Tensor forwardImpl(Tensor hidden_states) {
        hidden_states = hidden_states.transpose(1, 2);
        hidden_states = conv(hidden_states);
        hidden_states = padding(hidden_states);
        hidden_states = gelu(hidden_states);
        hidden_states = hidden_states.transpose(1, 2);
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

    Tensor shape(Tensor x, int64_t seq_len, int64_t bsz) {
        vector<int64_t> dim { bsz, seq_len, num_heads, head_dim };
        return x.view(dim).transpose(1, 2).contiguous();
    }
    
    Tensor forwardImpl(Tensor hidden_states) {

        // "Input shape: Batch x Time x Channel"
        auto bsz = hidden_states.sizes()[0];
        auto tgt_len = hidden_states.sizes()[1];

        //!!! why do we just reshape it twice? what is the point?
        Tensor query_states = q_proj(hidden_states) * scaling;
        Tensor key_states = shape(k_proj(hidden_states), -1, bsz);
        Tensor value_states = shape(v_proj(hidden_states), -1, bsz);

        cerr << "q = " << query_states.sizes() << ", k = " << key_states.sizes()
             << ", v = " << value_states.sizes() << endl;
        
        vector<int64_t> proj_shape { bsz * num_heads, -1, head_dim };
        query_states = shape(query_states, tgt_len, bsz).view(proj_shape);
        key_states = key_states.reshape(proj_shape);
        value_states = value_states.reshape(proj_shape);

        cerr << "q' = " << query_states.sizes() << ", k' = " << key_states.sizes()
             << ", v' = " << value_states.sizes() << endl;

        int64_t src_len = key_states.sizes()[1];
        Tensor attn_weights = bmm(query_states, key_states.transpose(1, 2));

        vector<int64_t> expected { bsz * num_heads, tgt_len, src_len };
        if (attn_weights.sizes() != expected) {
            cerr << "Attention weights should be of size " << expected
                 << " but are of size " << attn_weights.sizes() << endl;
        }

        // All masking etc omitted here (relevant only in training)
        
        attn_weights = softmax(attn_weights, -1);

        Tensor attn_output = bmm(attn_weights, value_states);
        
        expected = { bsz * num_heads, tgt_len, head_dim };
        if (attn_output.sizes() != expected) {
            cerr << "Attention output should be of size " << expected
                 << " but are of size " << attn_weights.sizes() << endl;
        }

        cerr << "output pre-reshape = " << attn_output.sizes() << endl;
        
        attn_output = attn_output.view({ bsz, num_heads, tgt_len, head_dim });

        cerr << "output after view(" << bsz << "," << num_heads << "," << tgt_len << "," << head_dim << ") = " << attn_output.sizes() << endl;
        
        attn_output = attn_output.transpose(1, 2);

        cerr << "output after transpose(1, 2) = " << attn_output.sizes() << endl;
        
        attn_output = attn_output.reshape({ bsz, tgt_len, embed_dim });

        cerr << "output after reshape(" << bsz << "," << tgt_len << "," << embed_dim << ") = " << attn_output.sizes() << endl;
        
        attn_output = out_proj(attn_output);
        
        cerr << "output after projection = " << attn_output.sizes() << endl;
        
        return attn_output;
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

    Tensor forwardImpl(Tensor hidden_states) {
        hidden_states = intermediate_dense(hidden_states);
        hidden_states = gelu(hidden_states);
        hidden_states = output_dense(hidden_states);
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

    Tensor forwardImpl(Tensor hidden_states) {
        Tensor attn_residual = hidden_states;
        hidden_states = attention(hidden_states);
        hidden_states = attn_residual + hidden_states;
        hidden_states = layer_norm(hidden_states);
        hidden_states = hidden_states + feed_forward(hidden_states);
        hidden_states = final_layer_norm(hidden_states);
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
            layers->push_back(register_module(to_string(i), HubertEncoderLayer()));
        }
    }

    vector<Tensor> forward(Tensor hidden_states) {
        
        Tensor position_embeddings = pos_conv_embed(hidden_states);

        hidden_states = hidden_states + position_embeddings;
        hidden_states = layer_norm(hidden_states);

        vector<Tensor> all_hidden_states;
        all_hidden_states.push_back(hidden_states);

        for (int i = 0; i < layers->size(); ++i) {
            if (auto layer = layers[i]->as<HubertEncoderLayer>()) {
                hidden_states = layer->forward(hidden_states);
                //!!! probably have to copy this explicitly
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

    vector<Tensor> forward(Tensor input_values) {
        Tensor extract_features = feature_extractor(input_values);
        extract_features = extract_features.transpose(1, 2);
        Tensor hidden_states = feature_projection(extract_features);
        auto encoder_outputs = encoder(hidden_states);
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
        cerr << i.key() << endl;
    }

    ifstream saved("for_libtorch.pth", ios::binary);
    saved.seekg(0, ios::end);
    auto size = saved.tellg();
    saved.seekg(0, ios::beg);

    cerr << "size = " << size << endl;
    if (size <= 0) {
        return 2;
    }
    
    vector<char> buffer(size);
    saved.read(buffer.data(), size);

    if (saved) {
        cerr << "read " << size << " chars" << endl;
    } else {
        cerr << "only read " << saved.gcount() << " of " << size << " chars"
             << endl;
        return 2;
    }
    
    auto obj = pickle_load(buffer);

    cerr << "Loaded the following:" << endl;
    auto dict = obj.toGenericDict();
    for (const auto &item : dict) {
        if (!item.key().isString()) {
            cerr << "(not a string key: " << item.key() << ")" << endl;
            continue;
        }
        string key = *(item.key().toString());
        auto value = item.value().toTensor();
        cerr << key << " -> " << value.sizes();
        if (params.contains(key)) {
            params[key].set_data(value);
            cerr << " - yes" << endl;
        } else {
            cerr << " - nope" << endl;
        }
    }

//    string testfile = "stairway-intro-16k-mono.wav";
    string testfile = "gerudo.wav";
    
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

    Tensor input = torch::from_blob(data.data(), { 1, 1, data.size() }).clone();

    vector<Tensor> output = mert(input);

    cerr << "received " << output.size() << " tensors as output" << endl;

    dump(output[0], "experiment-out-0");
    dump(output[12], "experiment-out-12");
}

