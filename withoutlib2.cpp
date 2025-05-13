
#ifndef LOCAL_NN_2_H
#define LOCAL_NN_2_H

#include <vector>
#include <variant>
#include <stdexcept>

namespace localnn2 {

typedef std::vector<float> t_1;
typedef std::vector<t_1> t_2;
typedef std::vector<t_2> t_3;
typedef std::vector<t_3> t_4;

typedef std::variant<t_1, t_2, t_3, t_4> Tensor;

static
std::vector<int64_t> sizes(const Tensor &t)
{
    if (const t_1 *s = std::get_if<t_1>(&t)) {
        return { (*s).size() };
    } else if (const t_2 *s = std::get_if<t_2>(&t)) {
        return { (*s).size(), (*s)[0].size() };
    } else if (const t_3 *s = std::get_if<t_3>(&t)) {
        return { (*s).size(), (*s)[0].size(), (*s)[0][0].size() };
    } else if (const t_4 *s = std::get_if<t_4>(&t)) {
        return { (*s).size(), (*s)[0].size(), (*s)[0][0].size(), (*s)[0][0][0].size() };
    } else {
        throw std::runtime_error("unsupported rank in sizes");
    }
}

static int64_t rank(const Tensor &t)
{
    if (std::holds_alternative<t_1>(t)) {
        return 1;
    } else if (std::holds_alternative<t_2>(t)) {
        return 2;
    } else if (std::holds_alternative<t_3>(t)) {
        return 3;
    } else if (std::holds_alternative<t_4>(t)) {
        return 4;
    } else {
        throw std::runtime_error("unsupported rank in rank");
    }
}

static int64_t numel(const Tensor &t)
{
    auto sz = sizes(t);
    int64_t n = 1;
    for (auto s : sz) n *= s;
    return n;
}

static
std::vector<float>
toData(const Tensor &t, std::vector<int64_t> &sz)
{
    sz = sizes(t);
    int64_t rank = sz.size();
    int64_t n = numel(t);
    std::vector<float> data(n, 0.f);

    switch (rank) {
    case 1: {
        data = std::get<t_1>(t);
        break;
    }
    case 2: {
        const t_2 &s2 = std::get<t_2>(t);
        for (int64_t i = 0; i < sz[0]; ++i) {
            std::copy(s2[i].begin(), s2[i].end(),
                      data.begin() + i * sz[1]);
        }
        break;
    }
    case 3: {
        const t_3 &s3 = std::get<t_3>(t);
        for (int64_t i = 0; i < sz[0]; ++i) {
            const t_2 &s2 = s3[i];
            for (int64_t j = 0; j < sz[1]; ++j) {
                std::copy(s2[j].begin(), s2[j].end(),
                          data.begin() + i * sz[1] * sz[2] + j * sz[2]);
            }
        }
        break;
    }
    case 4: {
        const t_4 &s4 = std::get<t_4>(t);
        for (int64_t i = 0; i < sz[0]; ++i) {
            const t_3 &s3 = s4[i];
            for (int64_t j = 0; j < sz[1]; ++j) {
                const t_2 &s2 = s3[j];
                for (int64_t k = 0; k < sz[2]; ++k) {
                    std::copy(s2[k].begin(), s2[k].end(),
                              data.begin() + i * sz[1] * sz[2] * sz[3]
                              + j * sz[2] * sz[3] + k * sz[3]);
                }
            }
        }
        break;
    }
    default:
        throw std::runtime_error("unsupported rank in toData");
    }

    return data;
}

template <typename T>
Tensor fromData(int64_t rank,
                const int64_t *sizes,
                const int64_t *strides,
                const T *const data) {

    Tensor result;
    
    switch (rank) {
    case 1: {
        t_1 t(sizes[0], 0.f);
        for (int64_t i = 0; i < sizes[0]; ++i) {
            t[i] = static_cast<float>(data[i * strides[0]]);
        }
        result = t;
        break;
    }
    case 2: {
        t_2 t;
        for (int64_t i = 0; i < sizes[0]; ++i) {
            Tensor s = fromData(rank - 1, sizes + 1, strides + 1,
                                data + i * strides[0]);
            t.push_back(std::get<t_1>(s));
        }
        result = t;
        break;
    }
    case 3: {
        t_3 t;
        for (int64_t i = 0; i < sizes[0]; ++i) {
            Tensor s = fromData(rank - 1, sizes + 1, strides + 1,
                                data + i * strides[0]);
            t.push_back(std::get<t_2>(s));
        }
        result = t;
        break;
    }
    case 4: {
        t_4 t;
        for (int64_t i = 0; i < sizes[0]; ++i) {
            Tensor s = fromData(rank - 1, sizes + 1, strides + 1,
                                data + i * strides[0]);
            t.push_back(std::get<t_3>(s));
        }
        result = t;
        break;
    }
    default:
        std::cerr << "rank = " << rank << std::endl;
        throw std::runtime_error("unsupported rank in fromData");
    }
    
    return result;
}



}

#endif
