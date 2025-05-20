
#ifndef LOCAL_NN_H
#define LOCAL_NN_H

#include <vector>

struct Tensor {
    typedef std::vector<int64_t> ivec;
    typedef std::vector<float> fvec;

    int64_t rank;
    ivec sizes;
    ivec strides;

    const float *data() const {
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
        return data()[index(i)];
    }
    inline float at(int64_t i, int64_t j) const {
        return data()[index(i, j)];
    }
    inline float at(int64_t i, int64_t j, int64_t k) const {
        return data()[index(i, j, k)];
    }
    inline float at(int64_t i, int64_t j, int64_t k, int64_t m) const {
        return data()[index(i, j, k, m)];
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


#endif
