
#ifndef LOCAL_NN_H
#define LOCAL_NN_H

#include <vector>

struct Tensor {
    typedef std::vector<int64_t> ivec;
    typedef std::vector<float> fvec;

    int64_t rank;
    ivec sizes;
    ivec strides;

    const float *constData() const {
        if (d_fixed) {
            return d_fixed;
        } else {
            return d_vec.data();
        }
    }

    float *mutableData() {
        if (d_fixed) {
            throw std::runtime_error("not mutable");
        } else {
            return d_vec.data();
        }
    }

    fvec copiedData() const {
        if (d_fixed) {
            return fvec(d_fixed, d_fixed + numel());
        } else {
            return d_vec;
        }
    }
    
    int64_t numel() const {
        return numelFor(sizes);
    }        

    /// Null
    Tensor() :
        Tensor(ivec()) {
    }
    
    /// Empty
    Tensor(ivec sizes_) :
        Tensor(sizes_, fvec(numelFor(sizes_), 0.f)) {
    }

    /// With data supplied
    Tensor(ivec sizes_, fvec data_) :
        rank(sizes_.size()),
        sizes(sizes_),
        strides(rank, 0),
        d_fixed(nullptr),
        d_vec(data_) {
        calcStrides();
        if (d_vec.size() != numel()) {
            throw std::runtime_error("size");
        }
    }

    /// With const data supplied
    static Tensor fromConst(ivec sizes_, const float *data_) {
        return Tensor(sizes_, data_);
    }

    inline int64_t index(int64_t i) const {
        return i * strides[0];
    }
    inline int64_t index(int64_t i, int64_t j) const {
        return i * strides[0] + j * strides[1];
    }
    inline int64_t index(int64_t i, int64_t j, int64_t k) const {
        return i * strides[0] + j * strides[1] + k * strides[2];
    }
    inline int64_t index(int64_t i, int64_t j, int64_t k, int64_t m) const {
        return i * strides[0] + j * strides[1] + k * strides[2] + m * strides[3];
    }

    inline float at(int64_t i) const {
        return constData()[index(i)];
    }
    inline float at(int64_t i, int64_t j) const {
        return constData()[index(i, j)];
    }
    inline float at(int64_t i, int64_t j, int64_t k) const {
        return constData()[index(i, j, k)];
    }
    inline float at(int64_t i, int64_t j, int64_t k, int64_t m) const {
        return constData()[index(i, j, k, m)];
    }

    Tensor &operator +=(const Tensor &t) {
        if (d_fixed) {
            throw std::runtime_error("not mutable");
        }
        int64_t n = numel();
        if (t.numel() != n) {
            throw std::runtime_error("size");
        }
        const float *tdata = t.constData();
#pragma GCC ivdep
        for (int64_t i = 0; i < n; ++i) {
            d_vec[i] += tdata[i];
        }
        return *this;
    }

    Tensor &operator *=(float x) {
        if (d_fixed) {
            throw std::runtime_error("not mutable");
        }
        int64_t n = numel();
        for (int64_t i = 0; i < n; ++i) {
            d_vec[i] *= x;
        }
        return *this;
    }        
    
    Tensor(const Tensor &t) =default;
    
    Tensor &operator=(const Tensor &t) {
        if (d_fixed) {
            throw std::runtime_error("not mutable");
        }
        rank = t.rank;
        sizes = t.sizes;
        strides = t.strides;
        d_vec = t.copiedData();
        return *this;
    }
    
private:
    const float *const d_fixed;
    fvec d_vec;

    Tensor(ivec sizes_, const float *data_) :
        rank(sizes_.size()),
        sizes(sizes_),
        d_fixed(data_) {
        calcStrides();
    }
    
    void calcStrides() {
        strides = ivec(sizes.size(), 1);
        int64_t s = 1;
        for (int64_t i = rank - 1; i >= 0; --i) {
            strides[i] = s;
            s *= sizes[i];
        }
    }
    
    static int64_t numelFor(const ivec &sizes_) {
        int64_t n = 1;
        for (auto i : sizes_) {
            n *= i;
        }
        return n;
    }
};

inline
std::ostream &
operator<<(std::ostream &out, const Tensor &t)
{
    out << "[";
    for (int64_t i = 0; i < t.rank; ++i) {
        out << t.sizes[i];
        if (i + 1 < t.rank) out << ", ";
    }
    out << "] {";
    for (int64_t i = 0; i < t.rank; ++i) {
        out << t.strides[i];
        if (i + 1 < t.rank) out << ", ";
    }
    out << "}";
    return out;
}

struct Ops
{
    static Tensor mm(const Tensor &a, const Tensor &b)
    {
        if (a.rank != 2 || b.rank != 2) {
            std::cerr << "unsupported rank (" << a.rank << " or "
                      << b.rank << ", should both be 2)" << std::endl;
            throw std::runtime_error("shape");
        }
        int64_t sa0 = a.sizes[0], sa1 = a.sizes[1];
        int64_t sb0 = b.sizes[0], sb1 = b.sizes[1];
        if (sa1 != sb0) {
            std::cerr << "incompatible sizes (" << sa1 << " != "
                 << sb0 << ")" << std::endl;
            throw std::runtime_error("shape");
        }
        Tensor out({ sa0, sb1 });
        const float *adata = a.constData();
        const float *bdata = b.constData();
        float *outdata = out.mutableData();
#pragma omp parallel for
        for (int64_t i = 0; i < sa0; ++i) {
            for (int64_t j = 0; j < sb1; ++j) {
                float x = 0.0;
                for (int64_t k = 0; k < sb0; ++k) {
                    x += adata[a.index(i, k)] * bdata[b.index(k, j)];
                }
                outdata[out.index(i, j)] = x;
            }
        }
        return out;
    }
    
    static Tensor bmm(const Tensor &t, const Tensor &m)
    {
        if (t.rank != 3 || m.rank != 3) {
            std::cerr << "unsupported rank (" << t.rank << " or "
                 << m.rank << ", should both be 3)" << std::endl;
            throw std::runtime_error("shape");
        }        
    
        Tensor out({ t.sizes[0], t.sizes[1], m.sizes[2] });

        if (t.sizes[2] != m.sizes[1]) {
            std::cerr << "incompatible sizes (" << t.sizes[2] << " != "
                 << m.sizes[1] << ")" << std::endl;
            throw std::runtime_error("shape");
        }

        float *outdata = out.mutableData();
    
        for (int64_t b = 0; b < t.sizes[0]; ++b) {
            Tensor tmp_t = Tensor::fromConst
                ({ t.sizes[1], t.sizes[2] }, t.constData() + t.index(b));
            Tensor tmp_m = Tensor::fromConst
                ({ m.sizes[1], m.sizes[2] }, m.constData() + m.index(b));
            Tensor tmp_out = mm(tmp_t, tmp_m);
            const float *tmpdata = tmp_out.constData();
            int64_t n = tmp_out.numel();
#pragma GCC ivdep
            for (int64_t i = 0; i < n; ++i) {
                outdata[out.index(b) + i] = tmpdata[i];
            }
        }

        return out;
    }

    static void linearImpl(const Tensor &in, int64_t inbase,
                           Tensor &out, int64_t outbase,
                           const Tensor &weight,
                           const Tensor &bias,
                           int64_t rank,
                           int64_t rix)
    {
        const float *indata = in.constData();
        const float *wdata = weight.constData();
        const float *bdata = bias.constData();
        float *outdata = out.mutableData();
        if (rix + 1 == rank) {
            const int64_t insize = in.sizes[rix];
            const int64_t outsize = out.sizes[rix];
#pragma omp parallel for
            for (int64_t j = 0; j < outsize; ++j) {
                float x = 0.f;
#pragma GCC ivdep
                for (int64_t i = 0; i < insize; ++i) {
                    x += wdata[j * insize + i] * indata[inbase + i];
                }
                x += bdata[j];
                outdata[outbase + j] = x;
            }
        } else {
#pragma omp parallel for
            for (int64_t i = 0; i < in.sizes[rix]; ++i) {
                linearImpl(in, inbase + i * in.strides[rix],
                           out, outbase + i * out.strides[rix],
                           weight, bias,
                           rank, rix + 1);
            }
        }
    }

    static Tensor linear(const Tensor &in,
                         const Tensor &weight,
                         const Tensor &bias)
    {
        auto rank = in.rank;
        auto outsizes = in.sizes;
        if (in.strides[rank-1] != 1) {
            throw std::runtime_error("not contiguous");
        }
        outsizes[rank-1] = weight.sizes[0];
        auto out = Tensor(outsizes);
        int rix = 0;
        while (in.sizes[rix] == 1) ++rix;
        linearImpl(in, 0,
                   out, 0,
                   weight, bias,
                   rank, rix);
        return out;
    }

    static void gelu(Tensor &t)
    {
        const double alpha = M_SQRT1_2;

        float *tdata = t.mutableData();
        for (int64_t i = 0; i < t.numel(); ++i) {
            double x = tdata[i];
            x = x * 0.5 * (1.0 + std::erf(x * alpha));
            tdata[i] = float(x);
        }
    }

    // We always copy for reshape and transpose, because we can only
    // handle contiguous layouts in some other functions

    static Tensor reshape(const Tensor &t, std::vector<int64_t> outsizes)
    {
        for (int i = 0; i < outsizes.size(); ++i) {
            if (outsizes[i] < 0) {
                outsizes[i] = t.numel();
                for (int j = 0; j < outsizes.size(); ++j) {
                    if (j != i && outsizes[j] > 0) {
                        outsizes[i] /= outsizes[j];
                    }
                }
                break;
            }
        }
        return Tensor(outsizes, t.copiedData());
    }

    static Tensor transpose12of3(const Tensor &t)
    {
        if (t.rank != 3) {
            throw std::runtime_error("shape");
        }
        int64_t a = t.sizes[0];
        int64_t b = t.sizes[1];
        int64_t c = t.sizes[2];
        std::vector<int64_t> outsizes = { a, c, b };
        Tensor out = Tensor(outsizes);
        const float *tdata = t.constData();
        float *outdata = out.mutableData();
        for (int64_t i = 0; i < a; ++i) {
            for (int64_t j = 0; j < b; ++j) {
#pragma GCC ivdep
                for (int64_t k = 0; k < c; ++k) {
                    outdata[out.index(i, k, j)] = tdata[t.index(i, j, k)];
                }
            }
        }
        return out;
    }

    static Tensor transpose12of4(const Tensor &t)
    {
        if (t.rank != 4) {
            throw std::runtime_error("shape");
        }
        int64_t a = t.sizes[0];
        int64_t b = t.sizes[1];
        int64_t c = t.sizes[2];
        int64_t d = t.sizes[3];
        std::vector<int64_t> outsizes = { a, c, b, d };
        Tensor out(outsizes);
        const float *tdata = t.constData();
        float *outdata = out.mutableData();
        for (int64_t i = 0; i < a; ++i) {
            for (int64_t j = 0; j < b; ++j) {
                for (int64_t k = 0; k < c; ++k) {
#pragma GCC ivdep
                    for (int64_t m = 0; m < d; ++m) {
                        outdata[out.index(i, k, j, m)] = tdata[t.index(i, j, k, m)];
                    }
                }
            }
        }
        return out;
    }

    static Tensor conv1d(const Tensor &t,
                         int64_t ch_in, int64_t ch_out, int64_t ksize,
                         int64_t stride, int64_t padding, int64_t groups,
                         const Tensor &w,
                         const Tensor *bp)
    {
        if (t.rank != 3 || t.strides[2] != 1) {
            throw std::runtime_error("unsupported format for input");
        }
        if (w.rank != 3 || w.strides[2] != 1) {
            throw std::runtime_error("unsupported format for weight");
        }
    
        // batchsize * inchannels * inlength -> batchsize * outchannels * outlength

        int64_t l_in = t.sizes[2];
        int64_t l_out = (l_in + 2 * padding - (ksize - 1) - 1) / stride + 1;

        Tensor out({ t.sizes[0], ch_out, l_out });
    
        const float *bbase = nullptr;
        if (bp) {
            bbase = bp->constData();
        }

        for (int64_t b = 0; b < t.sizes[0]; ++b) {
            for (int64_t g = 0; g < groups; ++g) {
                int c0 = g * (ch_out / groups);
                int k0 = g * (ch_in / groups);
                int g0 = g * ((out.numel() / t.sizes[0]) / groups);
#pragma omp parallel for
                for (int64_t c = 0; c < ch_out / groups; ++c) {

                    float *const outbase =
                        out.mutableData() + g0 + out.index(b, c);

                    for (int64_t k = 0; k < ch_in / groups; ++k) {
                        const float *const wbase =
                            w.constData() + w.index(c + c0, k);
                        const float *const tbase =
                            t.constData() + t.index(b, k + k0);
                        for (int64_t i = 0; i < ksize; ++i) {
                            if (padding == 0) {
#pragma GCC ivdep
                                for (int64_t x = 0; x < l_out; ++x) {
                                    outbase[x] += wbase[i] * tbase[x * stride + i];
                                }
                            } else {
                                // ew
                                for (int64_t x = 0; x < l_out; ++x) {
                                    int64_t x0 = x * stride + i - padding;
                                    if (x0 < 0 || x0 >= l_in) continue;
                                    outbase[x] += wbase[i] * tbase[x0];
                                }
                            }
                        }
                    }

                    if (bbase) {
#pragma GCC ivdep
                        for (int64_t x = 0; x < l_out; ++x) {
                            outbase[x] += bbase[c + c0];
                        }
                    }
                }
            }
        }

        return out;
    }

    static void softmax(Tensor &t)
    {
        int64_t h = *t.sizes.rbegin();
        int64_t m = t.numel() / h;

        for (int64_t j = 0; j < m; ++j) {

            float *base = t.mutableData() + j * h;

            double sum = 0.0;
            for (int64_t i = 0; i < h; ++i) {
                double x = exp(base[i]);
                sum += x;
                base[i] = x;
            }
            if (sum != 0.0) {
                for (int64_t i = 0; i < h; ++i) {
                    base[i] /= sum;
                }
            }
        }
    }

    static void layerNorm(Tensor &t,
                          const Tensor &weight, const Tensor &bias,
                          bool weightsPerInstance)
    {
        int64_t h = *t.sizes.rbegin();
        int64_t m = t.numel() / h;

        for (int64_t j = 0; j < m; ++j) {

            float *base = t.mutableData() + j * h;
        
            double mean = 0.0;
            for (int64_t i = 0; i < h; ++i) {
                mean += base[i];
            }
            mean /= double(h);

            double variance = 0.0;
            for (int64_t i = 0; i < h; ++i) {
                variance += (base[i] - mean) * (base[i] - mean);
            }
            variance /= double(h);

            double eps = 1.0e-5;
            double sd = sqrt(variance + eps);
            
            for (int64_t i = 0; i < h; ++i) {

                double x = base[i];
                double y = (x - mean) / sd;

                if (weightsPerInstance) {
                    // This case is for where the original model uses a
                    // GroupNorm layer. The GroupNorm is supplied with
                    // groups == channels so is actually performing an
                    // instance norm
                    y = y * weight.at(j) + bias.at(j);
                } else {
                    y = y * weight.at(i) + bias.at(i);
                }                

                base[i] = y;
            }
        }
    }
    
};

#endif
