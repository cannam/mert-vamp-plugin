
#include <iostream>
#include <cmath>
#include <memory>

#include "tensor.hpp"

#include "../data/weights.hpp"
#include "../data/parameters.hpp"

struct Module {
    virtual void prepare(std::string key) = 0;

    Tensor loadData(std::string keybase, std::string suffix) {
        std::string key = keybase;
        if (suffix != "") {
            key += "." + suffix;
        }
        if (key == "") {
            throw std::runtime_error("empty key");
        }
        if (key[0] == '.') {
            key = key.substr(1);
        }
        std::cerr << "Loading: " << key << " ... ";
        std::vector<int64_t> sizes;
        if (const float *data = lookup_model_data(key, sizes)) {
            std::cerr << "succeeded" << std::endl;
            return Tensor::fromConst(sizes, data);
        } else {
            std::cerr << "FAILED" << std::endl;
            throw std::runtime_error("failed to load data for key: " + key);
        }
    }
};

struct LayerBase : Module {
    virtual Tensor forward(const Tensor &x) = 0;
};

struct HubertNoLayerNormConvLayer : LayerBase {
    int64_t layerId = 0;
    Tensor weight;

    HubertNoLayerNormConvLayer(int64_t layerId_) : layerId(layerId_) { }

    void prepare(std::string key) override {
        weight = loadData(key, "conv.weight");
    }
    
    Tensor forward(const Tensor &x) override {

        int64_t inSize = 1;
        if (layerId > 0) inSize = convDimensions[layerId-1];
        int64_t outSize = convDimensions[layerId];

        auto tmp = Ops::conv1d(x, inSize, outSize,
                                convKernels[layerId], convStrides[layerId],
                                0, 1, weight, nullptr);
        Ops::gelu(tmp);
        
        return tmp;
    }
};

struct HubertGroupNormConvLayer : LayerBase {

    int64_t layerId = 0;
    Tensor convWeight;
    Tensor normWeight;
    Tensor normBias;
    
    HubertGroupNormConvLayer(int64_t layerId_) : layerId(layerId_) { }

    void prepare(std::string key) override {
        convWeight = loadData(key, "conv.weight");
        normWeight = loadData(key, "layer_norm.weight");
        normBias = loadData(key, "layer_norm.bias");
    }
    
    Tensor forward(const Tensor &x) override {

        int64_t inSize = 1;
        if (layerId > 0) inSize = convDimensions[layerId-1];
        int64_t outSize = convDimensions[layerId];

        auto tmp = Ops::conv1d(x, inSize, outSize,
                                convKernels[layerId], convStrides[layerId],
                                0, 1, convWeight, nullptr);

        Ops::layerNorm(tmp, normWeight, normBias, true);
        Ops::gelu(tmp);
        
        return tmp;
    }
};

struct HubertFeatureEncoder : LayerBase {

    std::vector<std::shared_ptr<LayerBase>> layers;

    HubertFeatureEncoder() {
        layers.push_back(std::make_shared<HubertGroupNormConvLayer>(0));
        for (int i = 1; i < convDimensions.size(); ++i) {
            layers.push_back(std::make_shared<HubertNoLayerNormConvLayer>(i));
        }
    }

    void prepare(std::string key) override {
        for (int i = 0; i < layers.size(); ++i) {
            layers[i]->prepare(key + ".conv_layers." + std::to_string(i));
        }
    }

    Tensor forward(const Tensor &x) {
        Tensor t = x;
        for (int i = 0; i < layers.size(); ++i) {
            t = layers[i]->forward(t);
        }
        return t;
    }
};

struct MERTFeatureProjection : LayerBase {

    Tensor normWeight;
    Tensor normBias;
    Tensor linearWeight;
    Tensor linearBias;
    
    MERTFeatureProjection() { }

    void prepare(std::string key) override {
        normWeight = loadData(key, "layer_norm.weight");
        normBias = loadData(key, "layer_norm.bias");
        linearWeight = loadData(key, "projection.weight");
        linearBias = loadData(key, "projection.bias");
    }

    Tensor forward(const Tensor &x) {
        Tensor t = x;
        Ops::layerNorm(t, normWeight, normBias, false);
        t = Ops::linear(t, linearWeight, linearBias);
        return t;
    }
};

struct HubertSamePadLayer : LayerBase {

    HubertSamePadLayer() { }

    void prepare(std::string) override { }
    
    Tensor forward(const Tensor &t) {
        if (nConvPosEmbeddings % 2 != 0) {
            return t;
        }
        if (t.rank != 3) {
            std::cerr << "error: wrong rank" << std::endl;
            throw std::runtime_error("shape");
        }
        auto outsizes = t.sizes;
        --(*outsizes.rbegin());
        auto out = Tensor(outsizes);
        auto tdata = t.constData();
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
    Tensor weight;
    Tensor bias;
    
    HubertPositionalConvEmbedding() { }

    void prepare(std::string key) override {
        // NB the PyTorch implementation uses a weight_norm
        // parameterisation - we don't have that here so when
        // exporting from PyTorch we must be sure to call
        // parametrize.remove_parametrizations on this layer first
        weight = loadData(key, "conv.weight");
        bias = loadData(key, "conv.bias");
        padding.prepare(key + ".padding");
    }

    Tensor forward(const Tensor &in) {
        auto hidden_states = Ops::transpose12of3(in);
        hidden_states = Ops::conv1d
            (hidden_states, hiddenSize, hiddenSize,
             nConvPosEmbeddings, 1, nConvPosEmbeddings/2,
             nConvPosEmbeddingGroups,
             weight, &bias);

        hidden_states = padding.forward(hidden_states);
        Ops::gelu(hidden_states);
        hidden_states = Ops::transpose12of3(hidden_states);
        return hidden_states;
    }
};

struct HubertAttention : LayerBase {

    Tensor kWeight;
    Tensor kBias;
    Tensor vWeight;
    Tensor vBias;
    Tensor qWeight;
    Tensor qBias;
    Tensor outWeight;
    Tensor outBias;
    
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

    void prepare(std::string key) override {
        kWeight = loadData(key, "k_proj.weight");
        kBias = loadData(key, "k_proj.bias");
        vWeight = loadData(key, "v_proj.weight");
        vBias = loadData(key, "v_proj.bias");
        qWeight = loadData(key, "q_proj.weight");
        qBias = loadData(key, "q_proj.bias");
        outWeight = loadData(key, "out_proj.weight");
        outBias = loadData(key, "out_proj.bias");
    }

    Tensor shape(const Tensor &x, int64_t seq_len, int64_t bsz) {
        std::vector<int64_t> dim { bsz, seq_len, num_heads, head_dim };
        return Ops::transpose12of4(Ops::reshape(x, dim));
    }
    
    Tensor forward(const Tensor &hidden_states) {

        std::cerr << "attention" << std::endl;
        
        auto bsz = hidden_states.sizes[0];
        auto tgt_len = hidden_states.sizes[1];

        auto query_states = Ops::linear(hidden_states, qWeight, qBias);

        query_states *= scaling;

        auto key_states =
            shape(Ops::linear(hidden_states, kWeight, kBias),
                  -1, bsz);

        auto value_states =
            shape(Ops::linear(hidden_states, vWeight, vBias),
                  -1, bsz);

        std::vector<int64_t> proj_shape { bsz * num_heads, -1, head_dim };
        query_states = Ops::reshape(shape(query_states, tgt_len, bsz), proj_shape);
        
        key_states = Ops::reshape(key_states, proj_shape);
        value_states = Ops::reshape(value_states, proj_shape);
        
        int64_t src_len = key_states.sizes[1];
        auto attn_weights = Ops::bmm(query_states, Ops::transpose12of3(key_states));

        std::vector<int64_t> expected { bsz * num_heads, tgt_len, src_len };
        if (attn_weights.sizes != expected) {
            throw std::runtime_error("shape");
        }

        Ops::softmax(attn_weights);

        auto attn_output = Ops::bmm(attn_weights, value_states);
        
        expected = { bsz * num_heads, tgt_len, head_dim };
        if (attn_output.sizes != expected) {
            throw std::runtime_error("shape");
        }

        attn_output = Ops::reshape(attn_output, { bsz, num_heads, tgt_len, head_dim });

        attn_output = Ops::transpose12of4(attn_output);
        attn_output = Ops::reshape(attn_output, { bsz, tgt_len, embed_dim });
        attn_output = Ops::linear(attn_output, outWeight, outBias);

        return attn_output;
    }
    
};

struct HubertFeedForward : LayerBase {

    Tensor intermediateWeight;
    Tensor intermediateBias;
    Tensor outputWeight;
    Tensor outputBias;

    HubertFeedForward() { }

    void prepare(std::string key) override {
        intermediateWeight = loadData(key, "intermediate_dense.weight");
        intermediateBias = loadData(key, "intermediate_dense.bias");
        outputWeight = loadData(key, "output_dense.weight");
        outputBias = loadData(key, "output_dense.bias");
    }

    Tensor forward(const Tensor &in) {

        auto hidden_states = Ops::linear
            (in, intermediateWeight, intermediateBias);

        Ops::gelu(hidden_states);
        
        hidden_states = Ops::linear
            (hidden_states, outputWeight, outputBias);
        
        return hidden_states;
    }
};


struct HubertEncoderLayer : LayerBase {

    HubertAttention attention;
    HubertFeedForward feed_forward;

    Tensor normWeight;
    Tensor normBias;
    Tensor finalNormWeight;
    Tensor finalNormBias;

    HubertEncoderLayer() { }

    void prepare(std::string key) override {
        normWeight = loadData(key, "layer_norm.weight");
        normBias = loadData(key, "layer_norm.bias");
        finalNormWeight = loadData(key, "final_layer_norm.weight");
        finalNormBias = loadData(key, "final_layer_norm.bias");
        attention.prepare(key + ".attention");
        feed_forward.prepare(key + ".feed_forward");
    }

    Tensor forward(const Tensor &in) {
        
        Tensor attn_residual = in;

        Tensor hidden_states = attention.forward(in);

        hidden_states += attn_residual;

        Ops::layerNorm(hidden_states, normWeight, normBias, false);

        Tensor ff = feed_forward.forward(hidden_states);

        hidden_states += ff;

        Ops::layerNorm(hidden_states, finalNormWeight, finalNormBias, false);
        
        return hidden_states;
    }
    
};

struct HubertEncoder : Module {

    HubertPositionalConvEmbedding pos_conv_embed;
    std::vector<std::shared_ptr<HubertEncoderLayer>> layers;

    Tensor normWeight;
    Tensor normBias;

    HubertEncoder() {
        for (int i = 0; i < nHiddenLayers; ++i) {
            layers.push_back(std::make_shared<HubertEncoderLayer>());
        }
    }

    void prepare(std::string key) override {
        normWeight = loadData(key, "layer_norm.weight");
        normBias = loadData(key, "layer_norm.bias");
        pos_conv_embed.prepare(key + ".pos_conv_embed");
        for (int i = 0; i < nHiddenLayers; ++i) {
            layers[i]->prepare(key + ".layers." + std::to_string(i));
        }
    }
    
    std::vector<Tensor> forward(const Tensor &in, int64_t rounds = -1) {

        auto hidden_states = pos_conv_embed.forward(in);

        hidden_states += in;

        Ops::layerNorm(hidden_states, normWeight, normBias, false);
        
        std::vector<Tensor> all_hidden_states;
        all_hidden_states.push_back(hidden_states);

        if (rounds < 0 || rounds > layers.size()) {
            rounds = layers.size();
        }
        
        for (int i = 0; i < rounds; ++i) {
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

    void prepare(std::string key) override {
        featureExtractor.prepare(key + ".feature_extractor");
        featureProjection.prepare(key + ".feature_projection");
        encoder.prepare(key + ".encoder");
    }

    std::vector<Tensor> forward(const Tensor &input_values, int rounds = -1) {
        auto features = featureExtractor.forward(input_values);
        features = Ops::transpose12of3(features);
        auto hidden_states = featureProjection.forward(features);
        auto encoder_outputs = encoder.forward(hidden_states, rounds);
        return encoder_outputs;
    }
};

