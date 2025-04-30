
//#include <torch/data.h>
//#include <torch/enum.h>
#include <torch/nn.h>
#include <torch/serialize.h>
//#include <torch/types.h>
//#include <torch/utils.h>

//#include <torch/torch.h>

#include <iostream>
#include <fstream>

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

struct HubertNoLayerNormConvLayerImpl : nn::Module {
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
    
    Tensor forward(Tensor x) {
        return gelu(conv(x));
    }
};

TORCH_MODULE(HubertNoLayerNormConvLayer);

struct HubertGroupNormConvLayerImpl : nn::Module {
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
    
    Tensor forward(Tensor x) {
        x = conv(x);
        x = layer_norm(x);
        return gelu(x);
    }
};

TORCH_MODULE(HubertGroupNormConvLayer);

struct HubertFeatureEncoderImpl : nn::Module {

    nn::ModuleList layers;
/*    
    HubertGroupNormConvLayer groupNormLayer = nullptr;
    vector<HubertNoLayerNormConvLayer> noLayerNormLayers;
*/
    HubertFeatureEncoderImpl() {
        layers = register_module("conv_layers", nn::ModuleList());
        layers->push_back(register_module("0", HubertGroupNormConvLayer(0)));
        for (int i = 1; i < convDimensions.size(); ++i) {
            layers->push_back(register_module(to_string(i), HubertNoLayerNormConvLayer(i)));
        }
    }

    Tensor forward(Tensor x) {
        for (int i = 0; i < layers->size(); ++i) {
            if (auto layer = layers[i]->as<HubertGroupNormConvLayer>()) {
                x = layer->forward(x);
            } else if (auto layer = layers[i]->as<HubertNoLayerNormConvLayer>()) {
                x = layer->forward(x);
            }
            return x;
        }
    }
};

TORCH_MODULE(HubertFeatureEncoder);

struct MERTImpl : nn::Module {
    
    HubertFeatureEncoder fe = nullptr;

    MERTImpl() {
        fe = register_module("feature_extractor", HubertFeatureEncoder());
    }

    Tensor forward(Tensor x) {
        return fe->forward(x);
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

    ifstream saved("fart.pth", ios::binary);
    saved.seekg(0, ios::end);
    auto size = saved.tellg();
    saved.seekg(0, ios::beg);

    cerr << "size = " << size << endl;
    if (size <= 0) {
        exit(2);
    }
    
    vector<char> buffer(size);
    saved.read(buffer.data(), size);

    if (saved) {
        cerr << "read " << size << " chars" << endl;
    } else {
        cerr << "only read " << saved.gcount() << " of " << size << " chars"
             << endl;
        exit(2);
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
        cerr << key << " -> " << value.sizes() << endl;
        if (params.contains(key)) {
            params[key].set_data(value);
        } else {
            cerr << "(didn't find this one)" << endl;
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
