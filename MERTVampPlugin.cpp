/*
    MERT Vamp Plugin
    Chris Cannam, Queen Mary University of London
    Copyright (c) 2025 Queen Mary University of London
*/

#include "MERTVampPlugin.h"
#include "Resampler.h"

#include "../data/weights.hpp"

#include "version.h"

#include <cmath>

using namespace std;

static float defaultChunkDuration = 8.f;
static int outputSampleRate = 50;

MERTVampPlugin::MERTVampPlugin(float inputSampleRate) :
    Plugin(inputSampleRate),
    m_blockSize(0),
    m_resampler(nullptr),
    m_chunkDuration(defaultChunkDuration),
    m_chunkLength(0)
{
}

MERTVampPlugin::~MERTVampPlugin()
{
    delete m_resampler;
}

string
MERTVampPlugin::getIdentifier() const
{
    return "mert-vamp-plugin";
}

string
MERTVampPlugin::getName() const
{
    return "MERT Vamp Plugin";
}

string
MERTVampPlugin::getDescription() const
{
    string desc = "Generates features from music audio using the MERT-v1-95M pretrained transformer model.";
#if defined(USE_LIBTORCH)
    desc += " This build uses LibTorch.";
#elif defined(USE_MKL)
    desc += " This build uses MKL.";
#elif defined(USE_ACCELERATE)
    desc += " This build uses the Accelerate framework.";
#else 
    desc += " This build does not use LibTorch or other frameworks and may be quite slow.";
#endif
    return desc;
}

string
MERTVampPlugin::getMaker() const
{
    return "QMUL";
}

int
MERTVampPlugin::getPluginVersion() const
{
    return PLUGIN_VERSION_INT;
}

string
MERTVampPlugin::getCopyright() const
{
    return "Code GPL; trained weights CC-BY-NC 4.0";
}

MERTVampPlugin::InputDomain
MERTVampPlugin::getInputDomain() const
{
    return TimeDomain;
}

size_t
MERTVampPlugin::getPreferredBlockSize() const
{
    return 0;
}

size_t 
MERTVampPlugin::getPreferredStepSize() const
{
    return 0;
}

size_t
MERTVampPlugin::getMinChannelCount() const
{
    return 1;
}

size_t
MERTVampPlugin::getMaxChannelCount() const
{
    return 1;
}

MERTVampPlugin::ParameterList
MERTVampPlugin::getParameterDescriptors() const
{
    ParameterList list;

    ParameterDescriptor d;
    d.identifier = "chunk";
    d.name = "Chunk duration";
    d.description = "Duration of the chunks into which the audio will be split before encoding. Longer chunks demand more memory to process.";
    d.unit = "s";
    d.minValue = 1;
    d.maxValue = 60;
    d.defaultValue = defaultChunkDuration;
    d.isQuantized = false;
    list.push_back(d);

    return list;
}

float
MERTVampPlugin::getParameter(string name) const
{
    if (name == "chunk") {
        return m_chunkDuration;
    }
    return 0;
}

void
MERTVampPlugin::setParameter(string name, float value) 
{
    if (name == "chunk") {
        m_chunkDuration = value;
    }
}

MERTVampPlugin::ProgramList
MERTVampPlugin::getPrograms() const
{
    ProgramList list;
    return list;
}

string
MERTVampPlugin::getCurrentProgram() const
{
    return ""; // no programs
}

void
MERTVampPlugin::selectProgram(string)
{
}

MERTVampPlugin::OutputList
MERTVampPlugin::getOutputDescriptors() const
{
    OutputList list;

    OutputDescriptor d;
    d.identifier = "conv";
    d.name = "Convolutional embedding";
    d.description = "Output from the convolutional preprocessor and positional embedding, as provided to the transformer as input.";
    d.unit = "";
    d.hasFixedBinCount = true;
    d.binCount = hiddenSize;
    d.hasKnownExtents = false; // for the conv output only
    d.isQuantized = false;
    d.sampleType = OutputDescriptor::FixedSampleRate;
    d.sampleRate = outputSampleRate;
    d.hasDuration = false;
    list.push_back(d);

    d.hasKnownExtents = true;
    d.minValue = -1.f;
    d.maxValue = 1.f;

    for (int i = 0; i < 12; ++i) {
        string is = to_string(i + 1);
        string is0 = is;
        if (i + 1 < 10) is0 = "0" + is0;
        d.identifier = "layer-" + is0;
        d.name = "Hidden layer " + is0 + " state";
        d.description = "Output from transformer layer " + is + ".";
        list.push_back(d);
    }
    
    return list;
}

bool
MERTVampPlugin::initialise(size_t channels, size_t stepSize, size_t blockSize)
{
    if (channels < getMinChannelCount()) {
        std::cerr << "MERTVampPlugin::initialise: unsupported channel count "
                  << channels << std::endl;
        return false;
    }

    if (blockSize != stepSize) {
        std::cerr << "MERTVampPlugin::initialise: block size " << blockSize
                  << " must match step size " << stepSize
                  << std::endl;
        return false;
    }

    if (blockSize == 0) {
        std::cerr << "MERTVampPlugin::initialise: zero block size!?"
                  << std::endl;
        return false;
    }

    bool firstTime = (m_blockSize == 0);
    
    m_channels = channels;
    m_blockSize = blockSize;

    if (firstTime) {
#ifdef USE_LIBTORCH
        m_mert->eval();
        auto params = m_mert->named_parameters();
        for (auto &p : params) {
            string key = p.key();
            vector<int64_t> sizes;
            if (auto data = lookup_model_data(key, sizes)) {
                // This seems hazardous - we don't want to clone because
                // we don't want to duplicate all the model data, but what
                // if it's in a protected const segment? Do we just hope
                // libtorch never tries to modify it?
                at::Tensor t = torch::from_blob(const_cast<float *>(data), sizes);
                params[key].set_data(t);
            }
        }
#else
        m_mert.prepare("");
#endif
    }
    
    reset();
    
    return true;
}

void
MERTVampPlugin::reset()
{
    delete m_resampler;
    m_resampler = nullptr;
    
    if (m_inputSampleRate != processingSampleRate) {
        m_resampler = new Resampler(m_inputSampleRate, processingSampleRate);
    }

    m_chunkLength = round(m_chunkDuration * processingSampleRate);
    
    m_chunk = {};
}

MERTVampPlugin::FeatureSet
MERTVampPlugin::process(const float *const *inputBuffers, Vamp::RealTime)
{
    FeatureSet fs;
    
    if (!m_chunkLength) {
        cerr << "ERROR: MERTVampPlugin::process: Not initialised" << endl;
        return fs;
    }

    vector<double> rin(m_blockSize, 0.0);
    for (int c = 0; c < m_channels; ++c) {
        for (int i = 0; i < m_blockSize; ++i) {
            rin[i] += inputBuffers[c][i];
        }
    }

    if (m_resampler) {
        auto rout = m_resampler->process(rin.data(), m_blockSize);
        m_chunk.insert(m_chunk.end(), rout.begin(), rout.end());
    } else {
        m_chunk.insert(m_chunk.end(), rin.begin(), rin.end());
    }        

    while (int64_t(m_chunk.size()) >= m_chunkLength) {
        processChunk(fs);
    }
    
    return fs;
}

void
MERTVampPlugin::processChunk(FeatureSet &fs)
{
    // If there is anything in the chunk, we should process it even if
    // it isn't complete - the caller makes that decision, not us. But
    // it could be longer than we need.

    int64_t length = m_chunkLength;
    vector<float> toProcess;
    
    if (int64_t(m_chunk.size()) < length) {
        toProcess = m_chunk;
        m_chunk = {};
    } else {
        toProcess = vector<float>(m_chunk.begin(), m_chunk.begin() + length);
        m_chunk = vector<float>(m_chunk.begin() + length, m_chunk.end());
    }
    
#ifdef USE_LIBTORCH
    at::Tensor input = torch::from_blob
        (chunk.data(), { 1, 1, int64_t(toProcess.size()) }); // no need to clone
    vector<at::Tensor> output = m_mert(input);
    for (int64_t i = 0; i < int64_t(output.size()); ++i) {
        at::Tensor t = output[i].to(at::kCPU).contiguous();
        const float *data = t.data_ptr<float>();
        auto sz = t.sizes();
        auto st = t.strides();
        for (int64_t j = 0; j < sz[0]; ++j) {
            for (int64_t k = 0; k < sz[1]; ++k) {
                
                Feature f;
                f.hasTimestamp = false;
                int64_t ix0 = j * st[0] + k * st[1];
                int64_t ix1 = j * st[0] + (k + 1) * st[1];
                f.values = vector<float>(data + ix0, data + ix1);
                fs[i].push_back(f);
            }
        }
    }
#else
    Tensor input({ 1, 1, int64_t(toProcess.size()) }, toProcess);
    vector<Tensor> output = m_mert.forward(input);
    for (int64_t i = 0; i < int64_t(output.size()); ++i) {
        Tensor &t = output[i];
        const float *data = t.constData();
        for (int64_t j = 0; j < t.sizes[0]; ++j) {
            for (int64_t k = 0; k < t.sizes[1]; ++k) {
                Feature f;
                f.hasTimestamp = false;
                int64_t ix0 = t.index(j, k);
                int64_t ix1 = t.index(j, k + 1);
                f.values = vector<float>(data + ix0, data + ix1);
                fs[i].push_back(f);
            }
        }
    }
#endif
}

MERTVampPlugin::FeatureSet
MERTVampPlugin::getRemainingFeatures()
{
    FeatureSet fs;
    
    while (int64_t(m_chunk.size()) >= m_chunkLength) {
        processChunk(fs);
    }
    if (m_chunk.size() > 0) {
        processChunk(fs);
    }
    
    return fs;
}


