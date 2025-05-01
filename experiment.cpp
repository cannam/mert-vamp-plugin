
//#include <torch/data.h>
//#include <torch/enum.h>

#include <torch/nn.h>
#include <torch/serialize.h>
//#include <torch/types.h>
//#include <torch/utils.h>

//#include <torch/torch.h>

#include <iostream>
#include <fstream>

#include <sndfile.h>

using namespace std;
using namespace torch;

static const int64_t vocabSize { 32 };
static const int64_t hiddenSize { 768 };
static const int64_t nHiddenLayers { 12 };
static const int64_t nAttentionHeads { 12 };
static const int64_t nConvPosEmbeddings { 128 };
static const int64_t nConfPosEmbeddingGroups { 16 };
static const int64_t intermediateSize { 3072 };
static const vector<int64_t> convDimensions { 512, 512, 512, 512, 512, 512, 512 };
static const vector<int64_t> convStrides { 5, 2, 2, 2, 2, 2, 2 };
static const vector<int64_t> convKernels { 10, 3, 3, 3, 3, 2, 2 };

struct LayerBase : nn::Module {
    virtual Tensor forward(Tensor x) = 0;
};

// important note: this is for inference only - I've omitted some
// logic that I believe is only used in training (dropout, certain
// types of masking)

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
    
    Tensor forward(Tensor x) override {
        cerr << "HubertNoLayerNormConvLayer: input shape = " << x.sizes() << endl;
        x = conv(x);
        cerr << "HubertNoLayerNormConvLayer: after conv = " << x.sizes() << endl;
        //!!! maybe it is nicer for the activations to be defined as separate layers rather than just ops
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
    
    Tensor forward(Tensor x) override {
        cerr << "HubertGroupNormConvLayer: input shape = " << x.sizes() << endl;
        x = conv(x);
        cerr << "HubertGroupNormConvLayer: after conv = " << x.sizes() << endl;
        x = layer_norm(x);
        cerr << "HubertGroupNormConvLayer: after norm = " << x.sizes() << endl;
        x = gelu(x);
        return x;
    }
};

TORCH_MODULE(HubertGroupNormConvLayer);

struct HubertFeatureEncoderImpl : nn::Module {

    nn::ModuleList layers;

    HubertFeatureEncoderImpl() {
        layers = register_module("conv_layers", nn::ModuleList());
        layers->push_back(register_module("0", HubertGroupNormConvLayer(0)));
        for (int i = 1; i < convDimensions.size(); ++i) {
            layers->push_back(register_module(to_string(i), HubertNoLayerNormConvLayer(i)));
        }
    }

    Tensor forward(Tensor x) {
        cerr << "HubertFeatureEncoder: input shape = " << x.sizes() << endl;
        for (int i = 0; i < layers->size(); ++i) {
            if (auto layer = layers[i]->as<LayerBase>()) {
                x = layer->forward(x);
                cerr << "HubertFeatureEncoder: after layer " << i << " = "
                     << x.sizes() << endl;
            } else {
                cerr << "HubertFeatureEncoder: Unexpected type for layer " << i
                     << endl;
            }
        }
        return x;
    }
};

TORCH_MODULE(HubertFeatureEncoder);

struct MERTFeatureProjectionImpl : nn::Module {

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

    Tensor forward(Tensor x) {
        cerr << "MERTFeatureProjection: input shape = " << x.sizes() << endl;
        x = layer_norm(x);
        cerr << "MERTFeatureProjection: after norm = " << x.sizes() << endl;
        x = linear(x);
        cerr << "MERTFeatureProjection: after linear = " << x.sizes() << endl;
        return x;
    }
        
};

TORCH_MODULE(MERTFeatureProjection);

struct HubertSamePadLayerImpl : nn::Module {

    //!!!
    HubertSamePadLayerImpl() { }
    
};

TORCH_MODULE(HubertSamePadLayer);

struct HubertPositionalConvEmbeddingImpl : nn::Module {

    nn::Conv1d conv = nullptr;
    HubertSamePadLayer padding = nullptr;
    
    //!!!
    HubertPositionalConvEmbeddingImpl() { }
    
};

TORCH_MODULE(HubertPositionalConvEmbedding);

struct HubertAttentionImpl : nn::Module {

    nn::Linear k_proj = nullptr;
    nn::Linear v_proj = nullptr;
    nn::Linear q_proj = nullptr;
    nn::Linear out_proj = nullptr;

    //!!!
    HubertAttentionImpl() { }
    
};

TORCH_MODULE(HubertAttention);

struct HubertFeedForwardImpl : nn::Module {

    nn::Linear intermediate_dense = nullptr;
    nn::Linear output_dense = nullptr;

    //!!!
    HubertFeedForwardImpl() { }
    
};

TORCH_MODULE(HubertFeedForward);


struct HubertEncoderLayerImpl : nn::Module {

    HubertAttention attention = nullptr;
    nn::LayerNorm layer_norm = nullptr;
    HubertFeedForward feed_forward = nullptr;
    nn::LayerNorm final_layer_norm = nullptr;

    //!!!
    HubertEncoderLayerImpl() { }
    
};

TORCH_MODULE(HubertEncoderLayer);

struct HubertEncoderImpl : nn::Module {

    HubertPositionalConvEmbedding pos_conv_embed = nullptr;
    nn::LayerNorm layer_norm = nullptr;
    nn::ModuleList layers;

    //!!!
    HubertEncoderImpl() { }

    
};

TORCH_MODULE(HubertEncoder);

struct MERTImpl : nn::Module {
    
    HubertFeatureEncoder fe = nullptr;
    MERTFeatureProjection proj = nullptr;
    HubertEncoder encoder = nullptr;

    MERTImpl() {
        fe = register_module("feature_extractor", HubertFeatureEncoder());
        proj = register_module("feature_projection", MERTFeatureProjection());
        encoder = register_module("encoder", HubertEncoder());
    }

    Tensor forward(Tensor x) {
        cerr << "MERT: input shape = " << x.sizes() << endl;
        x = fe(x);
        cerr << "MERT: after feature extractor = " << x.sizes() << endl;
        x = x.transpose(1, 2);
        cerr << "MERT: after transpose = " << x.sizes() << endl;
        x = proj(x);
        cerr << "MERT: after projection = " << x.sizes() << endl;
        //...
        return x;
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

    string testfile = "stairway-intro-16k-mono.wav";
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

    Tensor output = mert(input);

    // We did nothing to move it away from the CPU, but just in case
    // that changes later!
    Tensor result = output.to(kCPU);
    
    std::vector<float> v(result.data_ptr<float>(),
                         result.data_ptr<float>() + result.numel());

    ofstream csv("experiment-out.csv");
    int nrows = result.sizes()[1];
    int ncols = result.sizes()[2];
    cerr << "writing " << nrows << "-row " << ncols << "-column csv" << endl;
    for (int i = 0; i < nrows; ++i) {
        csv << i << ",";
        for (int j = 0; j < ncols; ++j) {
            csv << v[i * ncols + j];
            if (j + 1 < ncols) {
                csv << ",";
            } else {
                csv << endl;
            }
        }
    }
}


/*

struct TransformerEncoderLayerImpl : torch::nn::Module
{
    torch::nn::MultiheadAttention self_attn{nullptr};
    torch::nn::Linear linear1{nullptr}, linear2{nullptr};
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
    torch::nn::Dropout dropout{nullptr}, dropout1{nullptr}, dropout2{nullptr};

    TransformerEncoderLayerImpl(int64_t d_model, int64_t nhead,
                                int64_t dim_feedforward = 2048,
                                double dropout_prob = 0.1)
    {
        self_attn = register_module("self_attn", torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(d_model, nhead)));
        linear1 = register_module("linear1", torch::nn::Linear(d_model, dim_feedforward));
        linear2 = register_module("linear2", torch::nn::Linear(dim_feedforward, d_model));
        norm1 = register_module("norm1", torch::nn::LayerNorm(d_model));
        norm2 = register_module("norm2", torch::nn::LayerNorm(d_model));
        dropout = register_module("dropout", torch::nn::Dropout(dropout_prob));
        dropout1 = register_module("dropout1", torch::nn::Dropout(dropout_prob));
        dropout2 = register_module("dropout2", torch::nn::Dropout(dropout_prob));
    }

    torch::Tensor forward(torch::Tensor src) {
        // src: [T, B, D] (time-major)
        auto attn_output = std::get<0>(self_attn(src, src, src));  // Self-attention
        src = norm1(src + dropout1(attn_output));                 // Residual + Norm

        auto ff = dropout2(linear2(torch::relu(linear1(src))));  // Feedforward
        src = norm2(src + ff);                                    // Residual + Norm
        return src;
    }
};

TORCH_MODULE(TransformerEncoderLayer);

struct MyEncoderImpl : torch::nn::Module {
    std::vector<TransformerEncoderLayer> layers;

    MyEncoderImpl(int64_t num_layers, int64_t d_model, int64_t nhead) {
        for (int i = 0; i < num_layers; ++i) {
            layers.push_back(register_module("layer_" + std::to_string(i), TransformerEncoderLayer(d_model, nhead)));
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        for (auto& layer : layers) {
            x = layer->forward(x);
        }
        return x;
    }
};

TORCH_MODULE(MyEncoder);

*/
