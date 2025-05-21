
#include "model.hpp"

#include "../data/weights.hpp"

#include <iostream>
#include <fstream>

#include <sndfile.h>

using namespace std;
using namespace torch;

void dump(Tensor t, string filebase)
{
    // We assume t has channels as first dimension but only has one
    // channel, so we can treat as 2-d
    
    // We did nothing to move it away from the CPU, but just in case
    // that changes later!
    t = t.to(kCPU).contiguous();

    vector<float> v(t.data_ptr<float>(), t.data_ptr<float>() + t.numel());
    
    string filename = filebase + ".csv";
    ofstream csv(filename);

#ifdef DEBUG_TENSOR_SHAPES
    cerr << "will dump tensor of sizes " << t.sizes() << endl;
#endif
    
    int nrows = t.sizes()[1];
    int ncols = t.sizes()[2];

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

    cerr << "wrote " << nrows << "x" << ncols << " output to " << filename << endl;
}


int main(int argc, char **argv)
{
    MERT mert;
    mert->eval();

    auto params = mert->named_parameters();

    for (auto &p : params) {
        string key = p.key();
        vector<int64_t> sizes;
        if (auto data = lookup_model_data(key, sizes)) {
            // This seems hazardous - we don't want to clone because
            // we don't want to duplicate all the model data, but what
            // if it's in a protected const segment? Do we just hope
            // libtorch never tries to modify it?
            Tensor t = torch::from_blob(const_cast<float *>(data), sizes);
            params[key].set_data(t);
        }
    }

//    string testfile = "stairway-intro-16k-mono.wav";
    string testfile = "../data/gerudo.wav";
    
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

    Tensor input = torch::from_blob(data.data(), { 1, 1, int64_t(data.size()) }).clone();

    vector<Tensor> output = mert(input);

    cerr << "received " << output.size() << " tensors as output" << endl;

    dump(output[0], "experiment-out-0");
    dump(output[12], "experiment-out-12");
}

