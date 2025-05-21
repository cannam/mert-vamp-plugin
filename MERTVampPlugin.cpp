/*
    MERT Vamp Plugin
    Chris Cannam, Queen Mary University of London
    Copyright (c) 2025 Queen Mary University of London

    Permission is hereby granted, free of charge, to any person
    obtaining a copy of this software and associated documentation
    files (the "Software"), to deal in the Software without
    restriction, including without limitation the rights to use, copy,
    modify, merge, publish, distribute, sublicense, and/or sell copies
    of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be
    included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
    CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
    WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    Except as contained in this notice, the names of the Centre for
    Digital Music; Queen Mary, University of London; and Chris Cannam
    shall not be used in advertising or otherwise to promote the sale,
    use or other dealings in this Software without prior written
    authorization.
*/

#include "MERTVampPlugin.h"

#include "version.h"

#include <cmath>

using namespace std;

MERTVampPlugin::MERTVampPlugin(float inputSampleRate) :
    Plugin(inputSampleRate)
    /*!!!,
    m_blockSize(512),
    m_imageWidth(172),
    m_lossyCount(0),
    m_totalCount(0)
    */
{
}

MERTVampPlugin::~MERTVampPlugin()
{
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
    //!!! weights are CC-BY-NC 4.0
    return "MIT/X11 licence";
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
    return list;
}

float
MERTVampPlugin::getParameter(string) const
{
    return 0;
}

void
MERTVampPlugin::setParameter(string, float) 
{
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

    int outputNo = 0;
/*!!!
    OutputDescriptor d;
    d.identifier = "lossy";
    d.name = "Lossy";
    d.description = "A single estimate for whether the input has been lossily encoded in the past or not. 1 indicates yes it has, 0 indicates no it hasn't.";
    d.unit = "";
    d.hasFixedBinCount = true;
    d.binCount = 1;
    d.hasKnownExtents = true;
    d.minValue = 0.f;
    d.maxValue = 1.f;
    d.isQuantized = true;
    d.quantizeStep = 1.f;
    d.sampleType = OutputDescriptor::VariableSampleRate;
    d.hasDuration = true;
    m_lossyOutput = outputNo++;
    list.push_back(d);
*/
    return list;
}

bool
MERTVampPlugin::initialise(size_t channels, size_t stepSize, size_t blockSize)
{
    if (channels < getMinChannelCount() ||
	channels > getMaxChannelCount()) {
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

    return true;
}

void
MERTVampPlugin::reset()
{
    /*!!!
    m_buildingImage = {};
    m_lossyCount = 0;
    m_totalCount = 0;
    m_lastTimestamp = Vamp::RealTime::zeroTime;
    */
}

MERTVampPlugin::FeatureSet
MERTVampPlugin::process(const float *const *inputBuffers,
                       Vamp::RealTime timestamp)
{
    FeatureSet fs;
    

    return fs;
}

MERTVampPlugin::FeatureSet
MERTVampPlugin::getRemainingFeatures()
{
    FeatureSet fs;
    return fs;
}


