
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
static const int64_t nConvPosEmbeddingGroups { 16 };
static const int64_t intermediateSize { 3072 };
static const vector<int64_t> convDimensions { 512, 512, 512, 512, 512, 512, 512 };
static const vector<int64_t> convStrides { 5, 2, 2, 2, 2, 2, 2 };
static const vector<int64_t> convKernels { 10, 3, 3, 3, 3, 2, 2 };

struct LayerBase : nn::Module {
    virtual Tensor forward(Tensor x) = 0;
};

void dump(Tensor t, string filebase)
{
    // We assume t has channels as first dimension but only has one
    // channel, so we can treat as 2-d
    
    // We did nothing to move it away from the CPU, but just in case
    // that changes later!
    t = t.to(kCPU).contiguous();

    cerr << "layout = " << t.layout() << endl;
    cerr << "strides = " << t.strides() << endl;
    
    vector<float> v(t.data_ptr<float>(), t.data_ptr<float>() + t.numel());
    
    string filename = filebase + ".csv";
    ofstream csv(filename);
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
}

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

    HubertSamePadLayerImpl() { }

    Tensor forward(Tensor x) {
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

struct HubertPositionalConvEmbeddingImpl : nn::Module {

    nn::Conv1d conv = nullptr;
    HubertSamePadLayer padding = nullptr;
    
    HubertPositionalConvEmbeddingImpl() {

        // There doesn't *appear* to be a direct translation of
        // weight_norm in libtorch?

        // nb weight_norm is constructed with dim=2

        // See pytorch/aten/src/ATen/native/WeightNorm.cpp for most of
        // the implementation - perhaps we could even wangle calling it

        //!!! For now let's skip weight norm?
        
        nn::Conv1dOptions options =
            nn::Conv1dOptions(hiddenSize, hiddenSize, nConvPosEmbeddings)
            .padding(nConvPosEmbeddings/2)
            .groups(nConvPosEmbeddingGroups);

        conv = register_module("conv", nn::Conv1d(options));

        padding = register_module("padding", HubertSamePadLayer());

    }

    Tensor forward(Tensor hidden_states) {
        cerr << "HubertPositionalConvEmbedding: input shape = " << hidden_states.sizes() << endl;
        hidden_states = hidden_states.transpose(1, 2);
        cerr << "HubertPositionalConvEmbedding: after transpose = " << hidden_states.sizes() << endl;
        hidden_states = conv(hidden_states);
        cerr << "HubertPositionalConvEmbedding: after conv = " << hidden_states.sizes() << endl;
        hidden_states = padding(hidden_states);
        cerr << "HubertPositionalConvEmbedding: after padding = " << hidden_states.sizes() << endl;
        hidden_states = gelu(hidden_states);
        hidden_states = hidden_states.transpose(1, 2);
        cerr << "HubertPositionalConvEmbedding: returned shape = " << hidden_states.sizes() << endl;
        return hidden_states;
    }
    
};

TORCH_MODULE(HubertPositionalConvEmbedding);

struct HubertAttentionImpl : nn::Module {

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
    
    pair<Tensor, Tensor> forward(Tensor hidden_states) {
        cerr << "HubertAttention: input shape = " << hidden_states.sizes() << endl;

        // "Input shape: Batch x Time x Channel"
        auto bsz = hidden_states.sizes()[0];
        auto tgt_len = hidden_states.sizes()[1];
        //!!! why do we just reshape it twice? what is the point?
        Tensor query_states = q_proj(hidden_states) * scaling;
        Tensor key_states = shape(k_proj(hidden_states), -1, bsz);
        Tensor value_states = shape(v_proj(hidden_states), -1, bsz);
        
        cerr << "HubertAttention: initial query_states = " << query_states.sizes() << endl;
        cerr << "HubertAttention: initial key_states = " << key_states.sizes() << endl;
        cerr << "HubertAttention: initial value_states = " << value_states.sizes() << endl;

        vector<int64_t> proj_shape { bsz * num_heads, -1, head_dim };
        query_states = shape(query_states, tgt_len, bsz).view(proj_shape);
        key_states = key_states.reshape(proj_shape);
        value_states = value_states.reshape(proj_shape);

        cerr << "HubertAttention: subsequent query_states = " << query_states.sizes() << endl;
        cerr << "HubertAttention: subsequent key_states = " << key_states.sizes() << endl;
        cerr << "HubertAttention: subsequent value_states = " << value_states.sizes() << endl;

        int64_t src_len = key_states.sizes()[1];
        Tensor attn_weights = bmm(query_states, key_states.transpose(1, 2));

        cerr << "HubertAttention: attn_weights = " << attn_weights.sizes() << endl;
        
        vector<int64_t> expected { bsz * num_heads, tgt_len, src_len };
        if (attn_weights.sizes() != expected) {
            cerr << "Attention weights should be of size " << expected
                 << " but are of size " << attn_weights.sizes() << endl;
        }

        //!!! Attention mask -> ???, layer head mask -> apparently unused?
        // output_attentions -> false, dropout -> unused

        attn_weights = softmax(attn_weights, -1);

        Tensor attn_output = bmm(attn_weights, value_states);
        
        cerr << "HubertAttention: attn_output = " << attn_output.sizes() << endl;
        
        expected = { bsz * num_heads, tgt_len, head_dim };
        if (attn_output.sizes() != expected) {
            cerr << "Attention output should be of size " << expected
                 << " but are of size " << attn_weights.sizes() << endl;
        }

        vector<int64_t> out_shape { bsz, tgt_len, embed_dim };
        attn_output = attn_output.view(out_shape);

        attn_output = out_proj(attn_output);

        cerr << "HubertAttention: result attn_output = " << attn_output.sizes() << endl;
        cerr << "HubertAttention: result attn_weights = " << attn_weights.sizes() << endl;
        
        return { attn_output, attn_weights };
    }
    
};

TORCH_MODULE(HubertAttention);

struct HubertFeedForwardImpl : nn::Module {

    nn::Linear intermediate_dense = nullptr;
    nn::Linear output_dense = nullptr;

    HubertFeedForwardImpl() {
        intermediate_dense = register_module
            ("intermediate_dense", nn::Linear(hiddenSize, intermediateSize));
        output_dense = register_module
            ("output_dense", nn::Linear(intermediateSize, hiddenSize));
    }

    Tensor forward(Tensor hidden_states) {
        cerr << "HubertFeedForward: input shape = " << hidden_states.sizes() << endl;
        hidden_states = intermediate_dense(hidden_states);
        hidden_states = gelu(hidden_states);
        hidden_states = output_dense(hidden_states);
        cerr << "HubertFeedForward: returning shape = " << hidden_states.sizes() << endl;
        return hidden_states;
    }
    
};

TORCH_MODULE(HubertFeedForward);


struct HubertEncoderLayerImpl : nn::Module {

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

    Tensor forward(Tensor hidden_states) {
        cerr << "HubertEncoderLayer: input shape = " << hidden_states.sizes() << endl;
        Tensor attn_residual = hidden_states;
        auto attentionResult = attention(hidden_states);
        hidden_states = attentionResult.first;
        Tensor attn_weights = attentionResult.second;
        hidden_states = attn_residual + hidden_states;
        hidden_states = layer_norm(hidden_states);
        hidden_states = hidden_states + feed_forward(hidden_states);
        hidden_states = final_layer_norm(hidden_states);
        cerr << "HubertEncoderLayer: returning shape = " << hidden_states.sizes() << endl;
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
        dump(hidden_states, "mert-cpp-encoder-input");
        
        Tensor position_embeddings = pos_conv_embed(hidden_states);

        dump(position_embeddings, "mert-cpp-encoder-pos");

        hidden_states = hidden_states + position_embeddings;
        hidden_states = layer_norm(hidden_states);

        vector<Tensor> all_hidden_states;
        all_hidden_states.push_back(hidden_states);

        dump(hidden_states, "mert-cpp-encoder-prep");
        
        for (int i = 0; i < layers->size(); ++i) {
            if (auto layer = layers[i]->as<HubertEncoderLayer>()) {
                hidden_states = layer->forward(hidden_states);
                cerr << "HubertEncoder: after layer " << i << " = "
                     << hidden_states.sizes() << endl;
                //!!! probably have to copy this explicitly
                all_hidden_states.push_back(hidden_states);
                dump(hidden_states, "mert-cpp-encoder-" + to_string(i));
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
        cerr << "MERT: input shape = " << input_values.sizes() << endl;
        Tensor extract_features = feature_extractor(input_values);
        cerr << "MERT: after feature extractor = " << extract_features.sizes() << endl;
        extract_features = extract_features.transpose(1, 2);
        cerr << "MERT: after transpose = " << extract_features.sizes() << endl;
        dump(extract_features, "mert-cpp-features");
        Tensor hidden_states = feature_projection(extract_features);
        cerr << "MERT: after projection = " << hidden_states.sizes() << endl;
        dump(hidden_states, "mert-cpp-projection");
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

    int layerOfInterest = 12;
    
    // We did nothing to move it away from the CPU, but just in case
    // that changes later!
    Tensor result = output[layerOfInterest].to(kCPU);
    
    vector<float> v(result.data_ptr<float>(),
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
