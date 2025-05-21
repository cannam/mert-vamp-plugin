/*
    MERT Vamp Plugin
    Chris Cannam, Queen Mary University of London
    Copyright (c) 2025 Queen Mary University of London
*/

#include "MERTVampPlugin.h"

#include "data/parameters.hpp"

#include "ext/qm-dsp/dsp/rateconversion/Resampler.h"

#include "version.h"

#include <cmath>

using namespace std;

static float defaultChunkDuration = 8.f;
static bool defaultAdaptiveChunkStitching = true;
static int defaultTransformerRounds = nHiddenLayers;

MERTVampPlugin::MERTVampPlugin(float inputSampleRate) :
    Plugin(inputSampleRate),
    m_blockSize(0),
    m_resampler(nullptr),
    m_chunkDuration(defaultChunkDuration),
    m_adaptiveChunkStitching(defaultAdaptiveChunkStitching),
    m_transformerRounds(defaultTransformerRounds)
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
    return "Generates features from music audio using the MERT-v1-95M transformer model.";
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
    return 0;
}

MERTVampPlugin::ParameterList
MERTVampPlugin::getParameterDescriptors() const
{
    ParameterList list;

    ParameterDescriptor d;
    d.identifier = "chunk";
    d.name = "Chunk duration";
    d.description = "Duration of the chunks into which the audio will be split before encoding. Does not include any overlap required by the stitching method. Longer chunks demand more memory to process.";
    d.unit = "s";
    d.minValue = 1;
    d.maxValue = 60;
    d.defaultValue = defaultChunkDuration;
    d.isQuantized = false;
    list.push_back(d);

    d.identifier = "stitch";
    d.name = "Chunk stitching";
    d.description = "Method used to stitch chunks after processing. Adaptive means to add an overlap between chunks and choose a stitch position at a minimum difference between them. Naive means to make each chunk exactly the chunk duration.";
    d.unit = "";
    d.minValue = 0;
    d.maxValue = 1;
    d.defaultValue = (defaultAdaptiveChunkStitching ? 1 : 0);
    d.isQuantized = true;
    d.quantizeStep = 1;
    d.valueNames = { "Naive", "Adaptive" };
    list.push_back(d);

    d.identifier = "rounds";
    d.name = "Transformer rounds";
    d.description = "Number of rounds of the transformer architecture to run. This defines how many of the plugin's feature outputs contain valid data. Higher-numbered rounds typically correspond to higher-level musical features. If you reduce this from the default to some N, the plugin will run more quickly but only the first N hidden-layer outputs will contain any values. The default has all outputs populated.";
    d.unit = "rounds";
    d.minValue = 0;
    d.maxValue = nHiddenLayers;
    d.defaultValue = defaultTransformerRounds;
    d.isQuantized = true;
    d.quantizeStep = 1;
    d.valueNames = {};
    list.push_back(d);
    
    return list;
}

float
MERTVampPlugin::getParameter(string name) const
{
    if (name == "chunk") {
        return m_chunkDuration;
    } else if (name == "stitch") {
        return m_adaptiveChunkStitching ? 1.f : 0.f;
    } else if (name == "rounds") {
        return m_transformerRounds;
    }
    return 0;
}

void
MERTVampPlugin::setParameter(string name, float value) 
{
    if (name == "chunk") {
        m_chunkDuration = value;
    } else if (name == "stitch") {
        m_adaptiveChunkStitching = (value > 0.5f);
    } else if (name == "rounds") {
        m_transformerRounds = round(value);
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
    d.hasKnownExtents = true;
    d.minValue = -1.f;
    d.maxValue = 1.f;
    d.isQuantized = false;
    d.sampleType = OutputDescriptor::FixedSampleRate;
    d.sampleRate = 50.f;
    d.hasDuration = false;
    list.push_back(d);

    for (int i = 0; i < 12; ++i) {
        string is = to_string(i + 1);
        if (i + 1 < 10) is = "0" + is;
        d.identifier = "layer-" + is;
        d.name = "Hidden layer " + is + " state";
        d.description = "Output from transformer layer " + is + ". Will only be returned if the \"rounds\" parameter of the plugin is set to at least " + is + ".";
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

    m_channels = channels;
    m_blockSize = blockSize;
    
    reset();
    
    return true;
}

void
MERTVampPlugin::reset()
{
    delete m_resampler;
    m_resampler = new Resampler(m_inputSampleRate, 16000);
    m_chunk = {};
}

MERTVampPlugin::FeatureSet
MERTVampPlugin::process(const float *const *inputBuffers,
                        Vamp::RealTime timestamp)
{
    FeatureSet fs;

    vector<double> rin(m_blockSize, 0.0);
    for (int c = 0; c < m_channels; ++c) {
        for (int i = 0; i < m_blockSize; ++i) {
            rin[i] += inputBuffers[c][i];
        }
    }

    auto rout = m_resampler->process(rin.data(), m_blockSize);

    m_chunk.insert(m_chunk.end(), rout.begin(), rout.end());

    
    
    return fs;
}

MERTVampPlugin::FeatureSet
MERTVampPlugin::getRemainingFeatures()
{
    FeatureSet fs;
    return fs;
}


