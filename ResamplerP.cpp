/* -*- c-basic-offset: 4 indent-tabs-mode: nil -*-  vi:set ts=8 sts=4 sw=4: */

/*
    From the QM DSP Library
    Centre for Digital Music, Queen Mary, University of London.
*/

#include "ResamplerP.h"

#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>

using namespace std;

double
MathUtilities::factorial(int x)
{
    if (x < 0) return 0;
    double f = 1;
    for (int i = 1; i <= x; ++i) {
        f = f * i;
    }
    return f;
}

int
MathUtilities::gcd(int a, int b)
{
    int c = a % b;
    if (c == 0) {
        return b;
    } else {
        return gcd(b, c);
    }
}

KaiserWindow::Parameters
KaiserWindow::parametersForTransitionWidth(double attenuation,
                                           double transition)
{
    Parameters p;
    p.length = 1 + (attenuation > 21.0 ?
                    ceil((attenuation - 7.95) / (2.285 * transition)) :
                    ceil(5.79 / transition));
    p.beta = (attenuation > 50.0 ? 
              0.1102 * (attenuation - 8.7) :
              attenuation > 21.0 ? 
              0.5842 * pow(attenuation - 21.0, 0.4) + 0.07886 * (attenuation - 21.0) :
              0);
    return p;
}

static double besselTerm(double x, int i)
{
    if (i == 0) {
        return 1;
    } else {
        double f = MathUtilities::factorial(i);
        return pow(x/2, i*2) / (f*f);
    }
}

static double bessel0(double x)
{
    double b = 0.0;
    for (int i = 0; i < 20; ++i) {
        b += besselTerm(x, i);
    }
    return b;
}

void
KaiserWindow::init()
{
    double denominator = bessel0(m_beta);
    bool even = (m_length % 2 == 0);
    for (int i = 0; i < (even ? m_length/2 : (m_length+1)/2); ++i) {
        double k = double(2*i) / double(m_length-1) - 1.0;
        m_window.push_back(bessel0(m_beta * sqrt(1.0 - k*k)) / denominator);
    }
    for (int i = 0; i < (even ? m_length/2 : (m_length-1)/2); ++i) {
        m_window.push_back(m_window[int(m_length/2) - i - 1]);
    }
}

void
SincWindow::init()
{
    if (m_length < 1) {
        return;
    } else if (m_length < 2) {
        m_window.push_back(1);
        return;
    } else {

        int n0 = (m_length % 2 == 0 ? m_length/2 : (m_length - 1)/2);
        int n1 = (m_length % 2 == 0 ? m_length/2 : (m_length + 1)/2);
        double m = 2 * M_PI / m_p;

        for (int i = 0; i < n0; ++i) {
            double x = ((m_length / 2) - i) * m;
            m_window.push_back(sin(x) / x);
        }

        m_window.push_back(1.0);

        for (int i = 1; i < n1; ++i) {
            double x = i * m;
            m_window.push_back(sin(x) / x);
        }
    }
}

