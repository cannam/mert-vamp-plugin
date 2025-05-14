
#ifndef LOCAL_NN_H
#define LOCAL_NN_H

#include <vector>

namespace localnn {

struct Tensor {
    typedef std::vector<int64_t> ivec;
    typedef std::vector<float> fvec;

    int64_t rank;
    ivec sizes;
    ivec strides;
    fvec data;

    static int64_t numelFor(const ivec &sizes_) {
        int64_t n = 1;
        for (auto i : sizes_) {
            n *= i;
        }
        return n;
    }
    
    static Tensor empty(ivec sizes_) {
        auto n = numelFor(sizes_);
        int rank_ = sizes_.size();
        Tensor t { rank_,
                   sizes_,
                   ivec(rank_, 0),
                   fvec(n, 0.f) };
        int64_t s = 1;
        for (int64_t i = rank_ - 1; i >= 0; --i) {
            t.strides[i] = s;
            s *= t.sizes[i];
        }
        return t;
    }
    
    int64_t numel() const {
        int64_t acc = 1;
        for (auto i : sizes) {
            acc *= i;
        }
        return acc;
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
        return data[index(i)];
    }
    inline float at(int64_t i, int64_t j) const {
        return data[index(i, j)];
    }
    inline float at(int64_t i, int64_t j, int64_t k) const {
        return data[index(i, j, k)];
    }
    inline float at(int64_t i, int64_t j, int64_t k, int64_t m) const {
        return data[index(i, j, k, m)];
    }
};
    
template <typename T>
Tensor tensorFromData(int64_t rank_,
                      const int64_t *sizes_,
                      const int64_t *strides_,
                      const T *const data_) {
    int64_t n = 1;
    for (int64_t i = 0; i < rank_; ++i) {
        n *= sizes_[i];
    }
    Tensor t { rank_,
               Tensor::ivec(sizes_, sizes_ + rank_),
               Tensor::ivec(strides_, strides_ + rank_),
               Tensor::fvec(n, 0.f) };
    for (int64_t i = 0; i < n; ++i) {
        t.data[i] = static_cast<float>(data_[i]);
    }
    return t;
}
    
template <>
Tensor tensorFromData(int64_t rank_,
                      const int64_t *sizes_,
                      const int64_t *strides_,
                      const float *const data_) {
    int64_t n = 1;
    for (int64_t i = 0; i < rank_; ++i) {
        n *= sizes_[i];
    }
    Tensor t { rank_,
               Tensor::ivec(sizes_, sizes_ + rank_),
               Tensor::ivec(strides_, strides_ + rank_),
               Tensor::fvec(data_, data_ + n) };
    return t;
}

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

}

#endif
