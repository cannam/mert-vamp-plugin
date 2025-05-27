/* -*- c-basic-offset: 4 indent-tabs-mode: nil -*-  vi:set ts=8 sts=4 sw=4: */

/*
    From the QM DSP Library
    Centre for Digital Music, Queen Mary, University of London.
*/

#ifndef RESAMPLER_P_H
#define RESAMPLER_P_H

#include <vector>
#include <cmath>

#ifdef _MSC_VER
#define QM_R__ __restrict
#endif

#ifdef __GNUC__
#define QM_R__ __restrict__
#endif

#ifndef QM_R__
#define QM_R__
#endif

class MathUtilities  
{
public: 
    static double factorial(int x); // returns double in case it is large
    static int gcd(int a, int b);
};

class KaiserWindow
{
public:
    struct Parameters {
        int length;
        double beta;
    };

    /**
     * Construct a Kaiser windower with the given length and beta
     * parameter.
     */
    KaiserWindow(Parameters p) : m_length(p.length), m_beta(p.beta) { init(); }

    /**
     * Construct a Kaiser windower with the given attenuation in dB
     * and transition width in samples.
     */
    static KaiserWindow byTransitionWidth(double attenuation,
                                          double transition) {
        return KaiserWindow
            (parametersForTransitionWidth(attenuation, transition));
    }

    /**
     * Construct a Kaiser windower with the given attenuation in dB
     * and transition bandwidth in Hz for the given samplerate.
     */
    static KaiserWindow byBandwidth(double attenuation,
                                    double bandwidth,
                                    double samplerate) {
        return KaiserWindow
            (parametersForBandwidth(attenuation, bandwidth, samplerate));
    }

    /**
     * Obtain the parameters necessary for a Kaiser window of the
     * given attenuation in dB and transition width in samples.
     */
    static Parameters parametersForTransitionWidth(double attenuation,
                                                   double transition);

    /**
     * Obtain the parameters necessary for a Kaiser window of the
     * given attenuation in dB and transition bandwidth in Hz for the
     * given samplerate.
     */
    static Parameters parametersForBandwidth(double attenuation,
                                             double bandwidth,
                                             double samplerate) {
        return parametersForTransitionWidth
            (attenuation, (bandwidth * 2 * M_PI) / samplerate);
    } 

    int getLength() const {
        return m_length;
    }

    const double *getWindow() const { 
        return m_window.data();
    }

    void cut(double *src) const { 
        cut(src, src); 
    }

    void cut(const double *src, double *dst) const {
        for (int i = 0; i < m_length; ++i) {
            dst[i] = src[i] * m_window[i];
        }
    }

private:
    int m_length;
    double m_beta;
    std::vector<double> m_window;

    void init();
};

class SincWindow
{
public:
    /**
     * Construct a windower of the given length, containing the values
     * of sinc(x) with x=0 in the middle, i.e. at sample (length-1)/2
     * for odd or (length/2)+1 for even length, such that the distance
     * from -pi to pi (the nearest zero crossings either side of the
     * peak) is p samples.
     */
    SincWindow(int length, double p) : m_length(length), m_p(p) { init(); }

    int getLength() const {
        return m_length;
    }

    const double *getWindow() const { 
        return m_window.data();
    }

    void cut(double *src) const { 
        cut(src, src); 
    }

    void cut(const double *src, double *dst) const {
        for (int i = 0; i < m_length; ++i) {
            dst[i] = src[i] * m_window[i];
        }
    }

private:
    int m_length;
    double m_p;
    std::vector<double> m_window;

    void init();
};


#endif
