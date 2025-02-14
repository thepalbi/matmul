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

#ifndef M_MAX
#define M_MAX 4000
#endif

#ifndef K_MAX
#define K_MAX 4000
#endif

#ifndef N_MAX
#define N_MAX 4000
#endif

int main()
{
  namespace ublas = boost::numeric::ublas;

  const std::size_t m_min = 100;
  const std::size_t k_min = 100;
  const std::size_t n_min = 100;

  const std::size_t m_max = M_MAX;
  const std::size_t k_max = K_MAX;
  const std::size_t n_max = N_MAX;

  const std::size_t m_inc = 100;
  const std::size_t k_inc = 100;
  const std::size_t n_inc = 100;

  const bool matprodUpdate = true;

  typedef double T;
  typedef ublas::row_major SO;

  std::cout << "#   m";
  std::cout << "     n";
  std::cout << "     k";
  std::cout << "  uBLAS:   t1";
  std::cout << "       MFLOPS";
  std::cout << "   Blocked:   t2";
  std::cout << "      MFLOPS";
  std::cout << "        Diff nrm1";
  std::cout << std::endl;

  WallTime<double> walltime;

  for (std::size_t m = m_min, k = k_min, n = n_min;
       m <= m_max && k <= k_max && n <= n_max;
       m += m_inc, k += k_inc, n += n_inc)
  {
    ublas::matrix<T, SO> A(m, k);
    ublas::matrix<T, SO> B(k, n);
    ublas::matrix<T, SO> C1(m, n);
    ublas::matrix<T, SO> C2(m, n);

    fill(A);
    fill(B);
    fill(C1);
    C2 = C1;

    walltime.tic();
    ublas::axpy_prod(A, B, C1, matprodUpdate);
    double t1 = walltime.toc();

    walltime.tic();

    std::cout.width(5);
    std::cout << m << " ";
    std::cout.width(5);
    std::cout << n << " ";
    std::cout.width(5);
    std::cout << k << " ";
    std::cout.width(12);
    std::cout << t1 << " ";
    std::cout.width(12);
    std::cout << 2. * m / 1000. * n / 1000. * k / t1 << " ";
    std::cout.width(15);
    std::cout << std::endl;
  }
}