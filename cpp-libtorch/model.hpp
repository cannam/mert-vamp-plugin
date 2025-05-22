
#include <torch/nn.h>

#include <iostream>

#include "../data/parameters.hpp"

// For inference only - I've omitted some logic that I believe is only
// used in training (dropout, certain types of masking)

//#define DEBUG_TENSOR_SHAPES 1

#ifdef DEBUG_TENSOR_SHAPES
#include <cxxabi.h>
#endif

struct LayerBase : torch::nn::Module {
    virtual at::Tensor forwardImpl(at::Tensor x) = 0;
    
    virtual at::Tensor forward(at::Tensor x) {
#ifdef DEBUG_TENSOR_SHAPES
        auto inshape = x.sizes();
        auto instrides = x.strides();
#endif
        x = forwardImpl(x);
#ifdef DEBUG_TENSOR_SHAPES
        std::cerr << abi::__cxa_demangle(typeid(*this).name(), nullptr, nullptr, nullptr) << ": " << inshape << " (" << instrides << ") -> " << x.sizes() << " (" << x.strides() << ")" << std::endl;
#endif

        return x;
    }
};

struct HubertNoLayerNormConvLayerImpl : LayerBase {
    torch::nn::Conv1d conv = nullptr;

    HubertNoLayerNormConvLayerImpl(int64_t layerId) {
        int64_t inSize = 1;
        if (layerId > 0) inSize = convDimensions[layerId-1];
        int64_t outSize = convDimensions[layerId];

        torch::nn::Conv1dOptions options =
            torch::nn::Conv1dOptions(inSize, outSize, convKernels[layerId])
            .stride(convStrides[layerId]).bias(false);
        conv = register_module("conv", torch::nn::Conv1d(options));
    }
    
    at::Tensor forwardImpl(at::Tensor x) override {
        x = conv(x);
        x = gelu(x);
        return x;
    }
};

TORCH_MODULE(HubertNoLayerNormConvLayer);

struct HubertGroupNormConvLayerImpl : LayerBase {
    torch::nn::Conv1d conv = nullptr;
    torch::nn::GroupNorm layer_norm = nullptr;
    
    HubertGroupNormConvLayerImpl(int64_t layerId) {
        int64_t inSize = 1;
        if (layerId > 0) inSize = convDimensions[layerId-1];
        int64_t outSize = convDimensions[layerId];

        torch::nn::Conv1dOptions options =
            torch::nn::Conv1dOptions(inSize, outSize, convKernels[layerId])
            .stride(convStrides[layerId]).bias(false);
        conv = register_module("conv", torch::nn::Conv1d(options));

        torch::nn::GroupNormOptions normOptions =
            torch::nn::GroupNormOptions(outSize, outSize).affine(true);
        layer_norm = register_module("layer_norm", torch::nn::GroupNorm(normOptions));
    }
    
    at::Tensor forwardImpl(at::Tensor x) override {
        x = conv(x);
        x = layer_norm(x);
        x = gelu(x);
        return x;
    }
};

TORCH_MODULE(HubertGroupNormConvLayer);

struct HubertFeatureEncoderImpl : LayerBase {

    torch::nn::ModuleList layers;

    HubertFeatureEncoderImpl() {
        layers = register_module("conv_layers", torch::nn::ModuleList());
        layers->push_back(register_module("0", HubertGroupNormConvLayer(0)));
        for (int i = 1; i < convDimensions.size(); ++i) {
            layers->push_back(register_module(std::to_string(i), HubertNoLayerNormConvLayer(i)));
        }
    }

    at::Tensor forwardImpl(at::Tensor x) {
        for (int i = 0; i < layers->size(); ++i) {
            if (auto layer = layers[i]->as<LayerBase>()) {
                x = layer->forward(x);
            } else {
                std::cerr << "HubertFeatureEncoder: Unexpected type for layer " << i
                     << std::endl;
            }
        }
        return x;
    }
};

TORCH_MODULE(HubertFeatureEncoder);

struct MERTFeatureProjectionImpl : LayerBase {

    torch::nn::LayerNorm layer_norm = nullptr;
    torch::nn::Linear linear = nullptr;

    //!!! CQT would go here too but it doesn't seem to be in the
    //!!! config for the reference weights I have?
    
    MERTFeatureProjectionImpl() {
        int64_t inSize = *convDimensions.rbegin();
        int64_t outSize = hiddenSize;
        torch::nn::LayerNormOptions options({ inSize });
        layer_norm = register_module("layer_norm", torch::nn::LayerNorm(options));
        linear = register_module("projection", torch::nn::Linear(inSize, outSize));
    }

    at::Tensor forwardImpl(at::Tensor x) {
        x = layer_norm(x);
        x = linear(x);
        return x;
    }
        
};

TORCH_MODULE(MERTFeatureProjection);

struct HubertSamePadLayerImpl : LayerBase {

    HubertSamePadLayerImpl() { }

    at::Tensor forwardImpl(at::Tensor x) {
        if (nConvPosEmbeddings % 2 == 0) {
            // [:, :, : -1]
            return x.index({
                    torch::indexing::Slice(),
                    torch::indexing::Slice(),
                    torch::indexing::Slice(torch::indexing::None, -1)
                });
        } else {
            return x;
        }
    }
    
};

TORCH_MODULE(HubertSamePadLayer);

struct HubertPositionalConvEmbeddingImpl : LayerBase {

    torch::nn::Conv1d conv = nullptr;
    HubertSamePadLayer padding = nullptr;
    
    HubertPositionalConvEmbeddingImpl() {

        // The PyTorch implementation uses a weight_norm
        // parameterisation - we don't have that here so when
        // exporting from PyTorch we must be sure to call
        // parametrize.remove_parametrizations on this layer first
        
        torch::nn::Conv1dOptions options =
            torch::nn::Conv1dOptions(hiddenSize, hiddenSize, nConvPosEmbeddings)
            .padding(nConvPosEmbeddings/2)
            .groups(nConvPosEmbeddingGroups);
        conv = register_module("conv", torch::nn::Conv1d(options));
        padding = register_module("padding", HubertSamePadLayer());
    }

    at::Tensor forwardImpl(at::Tensor hidden_states) {
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

    torch::nn::Linear k_proj = nullptr;
    torch::nn::Linear v_proj = nullptr;
    torch::nn::Linear q_proj = nullptr;
    torch::nn::Linear out_proj = nullptr;

    int64_t embed_dim;
    int64_t num_heads;
    int64_t head_dim;
    double scaling;
    
    HubertAttentionImpl() :
        embed_dim(hiddenSize),
        num_heads(nAttentionHeads),
        head_dim(embed_dim / num_heads),
        scaling(pow(head_dim, -0.5)) {
        k_proj = register_module("k_proj", torch::nn::Linear(embed_dim, embed_dim));
        v_proj = register_module("v_proj", torch::nn::Linear(embed_dim, embed_dim));
        q_proj = register_module("q_proj", torch::nn::Linear(embed_dim, embed_dim));
        out_proj = register_module("out_proj", torch::nn::Linear(embed_dim, embed_dim));
    }

    at::Tensor shape(at::Tensor x, int64_t seq_len, int64_t bsz) {
        std::vector<int64_t> dim { bsz, seq_len, num_heads, head_dim };
        return x.view(dim).transpose(1, 2).contiguous();
    }
    
    at::Tensor forwardImpl(at::Tensor hidden_states) {
        
        // "Input shape: Batch x Time x Channel"
        auto bsz = hidden_states.sizes()[0];
        auto tgt_len = hidden_states.sizes()[1];

        at::Tensor query_states = q_proj(hidden_states) * scaling;
        at::Tensor key_states = shape(k_proj(hidden_states), -1, bsz);
        at::Tensor value_states = shape(v_proj(hidden_states), -1, bsz);

#ifdef DEBUG_TENSOR_SHAPES
        std::cerr << "q = " << query_states.sizes() << ", k = " << key_states.sizes()
             << ", v = " << value_states.sizes() << std::endl;
#endif
        
        std::vector<int64_t> proj_shape { bsz * num_heads, -1, head_dim };
        query_states = shape(query_states, tgt_len, bsz).view(proj_shape);
        key_states = key_states.reshape(proj_shape);
        value_states = value_states.reshape(proj_shape);

#ifdef DEBUG_TENSOR_SHAPES
        std::cerr << "q' = " << query_states.sizes() << ", k' = " << key_states.sizes()
             << ", v' = " << value_states.sizes() << std::endl;
#endif
        
        int64_t src_len = key_states.sizes()[1];
        at::Tensor attn_weights = bmm(query_states, key_states.transpose(1, 2));

        std::vector<int64_t> expected { bsz * num_heads, tgt_len, src_len };
        if (attn_weights.sizes() != expected) {
            std::cerr << "Attention weights should be of size " << expected
                 << " but are of size " << attn_weights.sizes() << std::endl;
        }

        // All masking etc omitted here (relevant only in training)
        
        attn_weights = softmax(attn_weights, -1);

        at::Tensor attn_output = bmm(attn_weights, value_states);
        
        expected = { bsz * num_heads, tgt_len, head_dim };
        if (attn_output.sizes() != expected) {
            std::cerr << "Attention output should be of size " << expected
                 << " but are of size " << attn_weights.sizes() << std::endl;
        }

#ifdef DEBUG_TENSOR_SHAPES
        std::cerr << "output pre-reshape = " << attn_output.sizes() << std::endl;
#endif
        
        attn_output = attn_output.view({ bsz, num_heads, tgt_len, head_dim });

#ifdef DEBUG_TENSOR_SHAPES
        std::cerr << "output after view(" << bsz << "," << num_heads << "," << tgt_len << "," << head_dim << ") = " << attn_output.sizes() << std::endl;
#endif
        
        attn_output = attn_output.transpose(1, 2);

#ifdef DEBUG_TENSOR_SHAPES
        std::cerr << "output after transpose(1, 2) = " << attn_output.sizes() << std::endl;
#endif
        
        attn_output = attn_output.reshape({ bsz, tgt_len, embed_dim });

#ifdef DEBUG_TENSOR_SHAPES
        std::cerr << "output after reshape(" << bsz << "," << tgt_len << "," << embed_dim << ") = " << attn_output.sizes() << std::endl;
#endif
        
        attn_output = out_proj(attn_output);
        
#ifdef DEBUG_TENSOR_SHAPES
        std::cerr << "output after projection = " << attn_output.sizes() << std::endl;
#endif
        
        return attn_output;
    }
    
};

TORCH_MODULE(HubertAttention);

struct HubertFeedForwardImpl : LayerBase {

    torch::nn::Linear intermediate_dense = nullptr;
    torch::nn::Linear output_dense = nullptr;

    HubertFeedForwardImpl() {
        intermediate_dense = register_module
            ("intermediate_dense", torch::nn::Linear(hiddenSize, intermediateSize));
        output_dense = register_module
            ("output_dense", torch::nn::Linear(intermediateSize, hiddenSize));
    }

    at::Tensor forwardImpl(at::Tensor hidden_states) {
        hidden_states = intermediate_dense(hidden_states);
        hidden_states = gelu(hidden_states);
        hidden_states = output_dense(hidden_states);
        return hidden_states;
    }
    
};

TORCH_MODULE(HubertFeedForward);


struct HubertEncoderLayerImpl : LayerBase {

    HubertAttention attention = nullptr;
    torch::nn::LayerNorm layer_norm = nullptr;
    HubertFeedForward feed_forward = nullptr;
    torch::nn::LayerNorm final_layer_norm = nullptr;

    HubertEncoderLayerImpl() {
        //!!! in the Python there is a choice of "eager" -> HubertAttention,
        // "sdpa" -> HubertSdpaAttention, "flash" -> HubertFlashAttention2.
        // The layer actually constructed in our example is "eager"
        attention = register_module("attention", HubertAttention());
        torch::nn::LayerNormOptions options({ hiddenSize });
        layer_norm = register_module("layer_norm", torch::nn::LayerNorm(options));
        feed_forward = register_module("feed_forward", HubertFeedForward());
        final_layer_norm = register_module("final_layer_norm", torch::nn::LayerNorm(options));
    }

    at::Tensor forwardImpl(at::Tensor hidden_states) {
        at::Tensor attn_residual = hidden_states;
        hidden_states = attention(hidden_states);
        hidden_states = attn_residual + hidden_states;
        hidden_states = layer_norm(hidden_states);
        hidden_states = hidden_states + feed_forward(hidden_states);
        hidden_states = final_layer_norm(hidden_states);
        return hidden_states;
    }
    
};

TORCH_MODULE(HubertEncoderLayer);

struct HubertEncoderImpl : torch::nn::Module {

    HubertPositionalConvEmbedding pos_conv_embed = nullptr;
    torch::nn::LayerNorm layer_norm = nullptr;
    torch::nn::ModuleList layers;

    HubertEncoderImpl() {
        pos_conv_embed = register_module
            ("pos_conv_embed", HubertPositionalConvEmbedding());
        torch::nn::LayerNormOptions options({ hiddenSize });
        layer_norm = register_module("layer_norm", torch::nn::LayerNorm(options));
        layers = register_module("layers", torch::nn::ModuleList());
        for (int i = 0; i < nHiddenLayers; ++i) {
            layers->push_back(register_module(std::to_string(i), HubertEncoderLayer()));
        }
    }

    std::vector<at::Tensor> forward(at::Tensor hidden_states) {
        
        at::Tensor position_embeddings = pos_conv_embed(hidden_states);

        hidden_states = hidden_states + position_embeddings;
        hidden_states = layer_norm(hidden_states);

        std::vector<at::Tensor> all_hidden_states;
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

struct MERTImpl : torch::nn::Module {
    
    HubertFeatureEncoder feature_extractor = nullptr;
    MERTFeatureProjection feature_projection = nullptr;
    HubertEncoder encoder = nullptr;

    MERTImpl() {
        feature_extractor = register_module("feature_extractor", HubertFeatureEncoder());
        feature_projection = register_module("feature_projection", MERTFeatureProjection());
        encoder = register_module("encoder", HubertEncoder());
    }

    std::vector<at::Tensor> forward(at::Tensor input_values) {
        at::Tensor extract_features = feature_extractor(input_values);
        extract_features = extract_features.transpose(1, 2);
        at::Tensor hidden_states = feature_projection(extract_features);
        auto encoder_outputs = encoder(hidden_states);
        return encoder_outputs;
    }
    
};

TORCH_MODULE(MERT);
