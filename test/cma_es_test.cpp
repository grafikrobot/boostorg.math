/*
 * Copyright Nick Thompson, 2024
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "math_unit_test.hpp"
#include "test_functions_for_optimization.hpp"
#include <boost/math/optimization/cma_es.hpp>
#include <boost/math/optimization/detail/multivariate_normal_distribution.hpp>
#include <array>
#include <algorithm>
#include <cmath>
#include <random>
#include <limits>

using std::abs;
using boost::math::optimization::cma_es;
using boost::math::optimization::cma_es_parameters;
using boost::math::optimization::detail::multivariate_normal_distribution;

template <class Real> void test_multivariate_normal() {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  constexpr const size_t n = 7;
  Matrix<Real, Dynamic, Dynamic> C = Matrix<Real, Dynamic, Dynamic>::Identity(n, n);
  std::mt19937_64 mt(12345);
  std::array<Real, n> mean;
  std::uniform_real_distribution<Real> dis(-1, 1);
  std::generate(mean.begin(), mean.end(), [&]() { return dis(mt); });
  auto mvn = multivariate_normal_distribution<decltype(mean)>(mean, C);
  std::array<Real, n> x;
  std::array<Real, n> empirical_means;
  empirical_means.fill(0);

  size_t i = 0;
  size_t samples = 2048;
  do {
    x = mvn(mt);
    for (size_t j = 0; j < n; ++j) {
      empirical_means[j] += x[j];
    }
  } while(i++ < samples);

  for (size_t j = 0; j < n; ++j) {
    empirical_means[j] /= samples;
    CHECK_ABSOLUTE_ERROR(mean[j], empirical_means[j], 0.05);
  }

  // Exhibits why we need to use the LDL^T decomposition:
  C = Matrix<Real, Dynamic, Dynamic>::Zero(n, n);
  mvn = multivariate_normal_distribution<decltype(mean)>(mean, C);
  i = 0;
  do {
    x = mvn(mt);
    for (size_t j = 0; j < n; ++j) {
      CHECK_EQUAL(mean[j], x[j]);
    }
  } while(i++ < 10);
  // Test that we're applying the permutation matrix correctly:
  C = Matrix<Real, Dynamic, Dynamic>::Zero(n, n);
  C(0,0) = 1;
  mvn = multivariate_normal_distribution<decltype(mean)>(mean, C);
  i = 0;
  do {
    x = mvn(mt);
    for (size_t j = 1; j < mean.size(); ++j) {
      CHECK_EQUAL(mean[j], x[j]);
    }
  } while(i++ < 3);

  C(0,0) = 0;
  C(n-1,n-1) = 1;
  mvn = multivariate_normal_distribution<decltype(mean)>(mean, C);
  i = 0;
  do {
    x = mvn(mt);
    // All but the last entry must be identical to the mean:
    for (size_t j = 0; j < mean.size() - 1; ++j) {
      CHECK_EQUAL(mean[j], x[j]);
    }
  } while(i++ < 3);

  C(0,0) = 0;
  C(1,1) = 1;
  C(n-1,n-1) = 0;
  mvn = multivariate_normal_distribution<decltype(mean)>(mean, C);
  i = 0;
  do {
    x = mvn(mt);
    for (size_t j = 0; j < mean.size() - 1; ++j) {
      if (j != 1) {
        CHECK_EQUAL(mean[j], x[j]);
      }
    }
  } while(i++ < 10);

  C(1,1) = 0;
  C(n-2,n-2) = 1;
  mvn = multivariate_normal_distribution<decltype(mean)>(mean, C);
  i = 0;
  do {
    x = mvn(mt);
    for (size_t j = 0; j < mean.size() - 1; ++j) {
      if (j != n-2) {
        CHECK_EQUAL(mean[j], x[j]);
      }
    }
  } while(i++ < 3);

  // Scaling test: If C->kC for some constant k, then A->sqrt(k)A.
  // First we build a random positive semidefinite matrix:
  Matrix<Real, Dynamic, Dynamic> C1 = Matrix<Real, Dynamic, Dynamic>::Random(n, n);
  C = C1.transpose()*C1;
  // Set the mean to 0:
  for (auto & m : mean) {
    m = 0;
  }
  samples = 1;
  std::vector<std::array<Real, n>> x1(samples);
  mt.seed(12859);
  mvn = multivariate_normal_distribution<decltype(mean)>(mean, C);
  for (size_t i = 0; i < x1.size(); ++i) {
    x1[i] = mvn(mt);
  }
  // Now scale C:
  C *= 16;
  // Set the seed back to the original:
  mt.seed(12859);
  std::vector<std::array<Real, n>> x2(samples);
  mvn = multivariate_normal_distribution<decltype(mean)>(mean, C);
  for (size_t i = 0; i < x2.size(); ++i) {
    x2[i] = mvn(mt);
  }
  // Now x2 = 4*x1 is expected:
  for (size_t i = 0; i < x1.size(); ++i) {
    for (size_t j = 0; j < n; ++j) {
      CHECK_ULP_CLOSE(4*x1[i][j], x2[i][j], 2);
    }
  }

}


template <class Real> void test_ackley() {
  std::cout << "Testing CMA-ES on Ackley function . . .\n";
  using ArgType = std::array<Real, 2>;
  auto params = cma_es_parameters<ArgType>();
  params.lower_bounds = {-5, -5};
  params.upper_bounds = {5, 5};

  std::mt19937_64 gen(12345);
  auto local_minima = cma_es(ackley<Real>, params, gen);
  CHECK_LE(std::abs(local_minima[0]), Real(0.1));
  CHECK_LE(std::abs(local_minima[1]), Real(0.1));

  // Does it work with a lambda?
  auto ack = [](std::array<Real, 2> const &x) { return ackley<Real>(x); };
  local_minima = cma_es(ack, params, gen);
  CHECK_LE(std::abs(local_minima[0]), Real(0.1));
  CHECK_LE(std::abs(local_minima[1]), Real(0.1));

  // Test that if an intial guess is the exact solution, the returned solution is the exact solution:
  std::array<Real, 2> initial_guess{0, 0};
  params.initial_guess = &initial_guess;
  local_minima = cma_es(ack, params, gen);
  CHECK_EQUAL(local_minima[0], Real(0));
  CHECK_EQUAL(local_minima[1], Real(0));

  std::atomic<bool> cancel = false;
  Real target_value = 0.0;
  std::atomic<Real> current_minimum_cost = std::numeric_limits<Real>::quiet_NaN();
  // Test query storage:
  std::vector<std::pair<ArgType, Real>> queries;
  local_minima = cma_es(ack, params, gen, target_value, &cancel, &current_minimum_cost, &queries);
  CHECK_EQUAL(local_minima[0], Real(0));
  CHECK_EQUAL(local_minima[1], Real(0));
  CHECK_LE(size_t(1), queries.size());
  for (auto const & q : queries) {
    auto expected = ackley<Real>(q.first);
    CHECK_EQUAL(expected, q.second);
  }
}


template <class Real> void test_rosenbrock_saddle() {
  std::cout << "Testing CMA-ES on Rosenbrock saddle . . .\n";
  using ArgType = std::array<Real, 2>;
  auto params = cma_es_parameters<ArgType>();
  params.lower_bounds = {0.5, 0.5};
  params.upper_bounds = {2.048, 2.048};
  params.max_function_calls = 20000;
  std::mt19937_64 gen(234568);
  auto local_minima = cma_es(rosenbrock_saddle<Real>, params, gen);

  CHECK_ABSOLUTE_ERROR(Real(1), local_minima[0], Real(0.05));
  CHECK_ABSOLUTE_ERROR(Real(1), local_minima[1], Real(0.05));

  // Does cancellation work?
  std::atomic<bool> cancel = true;
  gen.seed(12345);
  local_minima =
      cma_es(rosenbrock_saddle<Real>, params, gen, std::numeric_limits<Real>::quiet_NaN(), &cancel);
  CHECK_GE(std::abs(local_minima[0] - Real(1)), std::sqrt(std::numeric_limits<Real>::epsilon()));
}


template <class Real> void test_rastrigin() {
  std::cout << "Testing CMA-ES on Rastrigin function (global minimum = (0,0,...,0))\n";
  using ArgType = std::vector<Real>;
  auto params = cma_es_parameters<ArgType>();
  params.lower_bounds.resize(3, static_cast<Real>(-5.12));
  params.upper_bounds.resize(3, static_cast<Real>(5.12));
  params.max_function_calls = 1000000;
  std::mt19937_64 gen(34567);

  // By definition, the value of the function which a target value is provided must be <= target_value.
  Real target_value = 2.0;
  auto local_minima = cma_es(rastrigin<Real>, params, gen, target_value);
  CHECK_LE(rastrigin(local_minima), target_value);
}


// Tests NaN return types and return type != input type:
void test_sphere() {
  std::cout << "Testing CMA-ES on sphere . . .\n";
  using ArgType = std::vector<float>;
  auto params = cma_es_parameters<ArgType>();
  params.lower_bounds.resize(4, -1);
  params.upper_bounds.resize(4, 1);
  params.max_generations = 100000;
  std::mt19937_64 gen(56789);
  auto local_minima = cma_es(sphere, params, gen);
  for (auto x : local_minima) {
    CHECK_ABSOLUTE_ERROR(0.0f, x, 0.5f);
  }
}


template<typename Real>
void test_three_hump_camel() {
  std::cout << "Testing CMA-ES on three hump camel . . .\n";
  using ArgType = std::array<Real, 2>;
  auto params = cma_es_parameters<ArgType>();
  params.lower_bounds[0] = -5.0;
  params.lower_bounds[1] = -5.0;
  params.upper_bounds[0] = 5.0;
  params.upper_bounds[1] = 5.0;
  std::mt19937_64 gen(56789);
  auto local_minima = cma_es(three_hump_camel<Real>, params, gen);
  for (auto x : local_minima) {
    CHECK_ABSOLUTE_ERROR(0.0f, x, 0.2f);
  }
}


template<typename Real>
void test_beale() {
  std::cout << "Testing CMA-ES on the Beale function . . .\n";
  using ArgType = std::array<Real, 2>;
  auto params = cma_es_parameters<ArgType>();
  params.lower_bounds[0] = -5.0;
  params.lower_bounds[1] = -5.0;
  params.upper_bounds[0]= 5.0;
  params.upper_bounds[1]= 5.0;
  std::mt19937_64 gen(56789);
  auto local_minima = cma_es(beale<Real>, params, gen);
  CHECK_ABSOLUTE_ERROR(Real(3), local_minima[0], Real(0.1));
  CHECK_ABSOLUTE_ERROR(Real(1)/Real(2), local_minima[1], Real(0.1));
}

int main() {
#if 0 && (defined(__clang__) || defined(_MSC_VER))
  test_ackley<float>();
  test_ackley<double>();
  test_rosenbrock_saddle<double>();
  test_rastrigin<double>();
  test_three_hump_camel<float>();
  test_beale<double>();
#endif
  test_sphere();
  test_multivariate_normal<double>();
  return boost::math::test::report_errors();
}
