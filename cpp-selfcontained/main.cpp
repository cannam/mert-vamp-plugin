
#include <iostream>
#include <fstream>
#include <cmath>
#include <memory>

#include <sndfile.h>

#include "model.hpp"

using namespace std;

void dump(const Tensor &tt, string filebase)
{
    string filename = filebase + ".csv";
    ofstream csv(filename);

    int base = tt.sizes.size() - 2;
    int nrows = 1;
    
    if (base < -1 || base > 2) {
        cerr << "unsupported shape in dump";
        exit(2);
    }
    if (base >= 0) {
        nrows = tt.sizes[base + 0];
    }

    int ncols = tt.sizes[base + 1];

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
            switch (base) {
            case -1: csv << tt.at(j); break;
            case 0: csv << tt.at(i, j); break;
            case 1: csv << tt.at(0, i, j); break;
            case 2: csv << tt.at(0, 0, i, j); break;
            }
                
            if (j + 1 < ncols) {
                csv << ",";
            } else {
                csv << endl;
            }
        }
    }
}

int main(int argc, char **argv)
{
    MERT mert;

    mert.prepare("");

    string testfile = "../data/testfile.wav";
    
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

    Tensor input({ 1, 1, int64_t(data.size()) }, data);
    
    vector<Tensor> output = mert.forward(input);

    cerr << "received " << output.size() << " tensors as output" << endl;

    dump(output[0], "experiment-out-0");
    dump(output[12], "experiment-out-12");
}

