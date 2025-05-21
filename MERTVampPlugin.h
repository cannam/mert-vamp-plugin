/*
    MERT Vamp Plugin
    Chris Cannam, Queen Mary University of London
    Copyright (c) 2025 Queen Mary University of London
*/

#ifndef MERT_VAMP_PLUGIN_H
#define MERT_VAMP_PLUGIN_H

#include <vamp-sdk/Plugin.h>

class Resampler;

class MERTVampPlugin : public Vamp::Plugin
{
public:
    MERTVampPlugin(float inputSampleRate);
    virtual ~MERTVampPlugin();

    std::string getIdentifier() const;
    std::string getName() const;
    std::string getDescription() const;
    std::string getMaker() const;
    int getPluginVersion() const;
    std::string getCopyright() const;

    InputDomain getInputDomain() const;
    size_t getPreferredBlockSize() const;
    size_t getPreferredStepSize() const;
    size_t getMinChannelCount() const;
    size_t getMaxChannelCount() const;

    ParameterList getParameterDescriptors() const;
    float getParameter(std::string identifier) const;
    void setParameter(std::string identifier, float value);

    ProgramList getPrograms() const;
    std::string getCurrentProgram() const;
    void selectProgram(std::string name);

    OutputList getOutputDescriptors() const;

    bool initialise(size_t channels, size_t stepSize, size_t blockSize);
    void reset();

    FeatureSet process(const float *const *inputBuffers,
                       Vamp::RealTime timestamp);

    FeatureSet getRemainingFeatures();

protected:
    int m_channels;
    int m_blockSize;
    Resampler *m_resampler;
    
    float m_chunkDuration;
    bool m_adaptiveChunkStitching;
    int m_transformerRounds;

    std::vector<float> m_chunk;
};



#endif
