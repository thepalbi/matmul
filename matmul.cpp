#include <cassert>
#include <chrono>
#include <cmath>
#include <random>
#include <boost/timer.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/operation.hpp>

// matmul example copied from: https://www.mathematik.uni-ulm.de/~lehn/test_ublas/session1/page01.html

template <typename T>
struct WallTime
{
  void tic()
  {
    t0 = std::chrono::high_resolution_clock::now();
  }

  T toc()
  {
    using namespace std::chrono;

    elapsed = high_resolution_clock::now() - t0;
    return duration<T, seconds::period>(elapsed).count();
  }

  std::chrono::high_resolution_clock::time_point t0;
  std::chrono::high_resolution_clock::duration elapsed;
};

template <typename MATRIX>
void fill(MATRIX &A)
{
  typedef typename MATRIX::size_type size_type;
  typedef typename MATRIX::value_type T;

  std::random_device random;
  std::default_random_engine mt(random());
  std::uniform_real_distribution<T> uniform(-100, 100);

  for (size_type i = 0; i < A.size1(); ++i)
  {
    for (size_type j = 0; j < A.size2(); ++j)
    {
      A(i, j) = uniform(mt);
    }
  }
}

#ifndef N_MAX
#define N_MAX 1000
#endif

int main()
{
  namespace ublas = boost::numeric::ublas;

  const std::size_t n_min = 100;
  const std::size_t n_max = N_MAX;
  const std::size_t n_inc = 100;

  typedef double T;
  typedef ublas::row_major SO;

  WallTime<double> walltime;

  for (std::size_t n = n_min; n <= n_max; n += n_inc)
  {
    ublas::matrix<T, SO> A(n, n);
    ublas::matrix<T, SO> B(n, n);
    ublas::matrix<T, SO> C1(n, n);

    fill(A);
    fill(B);
    fill(C1);

    walltime.tic();
    ublas::axpy_prod(A, B, C1, true);
    double t1 = walltime.toc();

    std::cout << n << " " << t1 << std::endl;
  }
}