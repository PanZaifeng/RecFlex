#pragma once
#include <cub/cub.cuh>


// Modified from AlignedVector of TensorFlow 2.6
// Represents an aligned array of N elements of T. Data pointers can be
// reinterpreted as this type to generate vectorized loads/stores in a kernel.
template <typename T, int N> struct alignas(alignof(T) * N) AlignedVector {
public:
  typedef T value_type;
  static constexpr const int kSize = N;

  AlignedVector() = default;

  // Uniform initialization.
  __host__ __device__ explicit AlignedVector(value_type uniform) {
#pragma unroll
    for (int i = 0; i < kSize; ++i) {
      values_[i] = uniform;
    }
  }
  // Uniform initialization with explicit conversion.
  // Note: This is required for T=Eigen::half because it only supports explicit
  // conversions from other types and its template constructor is too relaxed
  // to be able to use std::is_constructible.
  template <typename U, typename std::enable_if<std::is_arithmetic<U>::value,
                                                int>::type = 0>
  __host__ __device__ explicit AlignedVector(U uniform_u) {
    value_type uniform(uniform_u);
#pragma unroll
    for (int i = 0; i < kSize; ++i) {
      values_[i] = uniform;
    }
  }
  // Implicit conversion.
  template <typename U, typename std::enable_if<
                            std::is_convertible<U, T>::value, int>::type = 0>
  __host__ __device__ AlignedVector(const AlignedVector<U, N> &other) {
#pragma unroll
    for (int i = 0; i < kSize; ++i) {
      values_[i] = other[i];
    }
  }
  // Explicit conversion.
  template <typename U,
            typename std::enable_if<!std::is_convertible<U, T>::value &&
                                        std::is_constructible<T, U>::value,
                                    int>::type = 0>
  __host__ __device__ explicit AlignedVector(const AlignedVector<U, N> &other) {
#pragma unroll
    for (int i = 0; i < kSize; ++i) {
      values_[i] = T(other[i]);
    }
  }

  __host__ __device__ value_type &operator[](int i) { return values_[i]; }
  __host__ __device__ const value_type &operator[](int i) const {
    return values_[i];
  }

#define DEFINE_BINARY_UPDATE_OPERATOR(op)                                      \
  __host__ __device__ AlignedVector &operator op(const AlignedVector & rhs) {  \
    _Pragma("unroll") for (int i = 0; i < kSize; ++i) {                        \
      values_[i] op rhs[i];                                                    \
    }                                                                          \
    return *this;                                                              \
  }                                                                            \
  __host__ __device__ AlignedVector &operator op(const T & rhs) {              \
    _Pragma("unroll") for (int i = 0; i < kSize; ++i) { values_[i] op rhs; }   \
    return *this;                                                              \
  }
  DEFINE_BINARY_UPDATE_OPERATOR(+=)
  DEFINE_BINARY_UPDATE_OPERATOR(-=)
  DEFINE_BINARY_UPDATE_OPERATOR(*=)
  DEFINE_BINARY_UPDATE_OPERATOR(/=)
#undef DEFINE_BINARY_UPDATE_OPERATOR

#define DEFINE_BINARY_OPERATOR(op)                                             \
  friend __host__ __device__ AlignedVector operator op(                        \
      const AlignedVector &lhs, const AlignedVector &rhs) {                    \
    AlignedVector ret;                                                         \
    _Pragma("unroll") for (int i = 0; i < kSize; ++i) {                        \
      ret[i] = lhs[i] op rhs[i];                                               \
    }                                                                          \
    return ret;                                                                \
  }                                                                            \
  friend __host__ __device__ AlignedVector operator op(                        \
      const AlignedVector &lhs, const T &rhs) {                                \
    AlignedVector ret;                                                         \
    _Pragma("unroll") for (int i = 0; i < kSize; ++i) {                        \
      ret[i] = lhs[i] op rhs;                                                  \
    }                                                                          \
    return ret;                                                                \
  }
  DEFINE_BINARY_OPERATOR(+)
  DEFINE_BINARY_OPERATOR(-)
  DEFINE_BINARY_OPERATOR(*)
  DEFINE_BINARY_OPERATOR(/)
#undef DEFINE_BINARY_OPERATOR

  __host__ __device__ void LoadFrom(const T *__restrict__ src, int vector_idx,
                                    int limit, int default_value) {
    int begin_idx = vector_idx * kSize;
    int num_rest = limit - begin_idx;
    if (num_rest >= kSize) {
      *this = reinterpret_cast<const AlignedVector<T, N> *>(src)[vector_idx];
    } else if (num_rest <= 0) {
      *this = AlignedVector<T, N>(default_value);
    } else {
#pragma unroll
      for (int i = 0; i < kSize; ++i) {
        values_[i] = i < num_rest ? src[begin_idx + i] : default_value;
      }
    }
  }

  __host__ __device__ void StoreTo(T *__restrict__ dst, int vector_idx,
                                   int limit) {
    int begin_idx = vector_idx * kSize;
    int num_rest = limit - begin_idx;
    if (num_rest >= kSize) {
      reinterpret_cast<AlignedVector<T, N> *>(dst)[vector_idx] = *this;
    } else if (num_rest <= 0) {
      // pass
    } else {
#pragma unroll
      for (int i = 0; i < kSize; ++i) {
        if (i < num_rest)
          dst[begin_idx + i] = values_[i];
      }
    }
  }

public:
  value_type values_[N];
};
