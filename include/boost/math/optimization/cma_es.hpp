/*
 * Copyright Nick Thompson, 2024
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */
#ifndef BOOST_MATH_OPTIMIZATION_CMA_ES_HPP
#define BOOST_MATH_OPTIMIZATION_CMA_ES_HPP
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cassert>
#include <limits>
#include <list>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#include <boost/math/optimization/detail/common.hpp>
#if __has_include(<Eigen/Dense>)
#include <boost/math/optimization/detail/multivariate_normal_distribution.hpp>
#include <Eigen/Dense>
#else
#error "CMA-ES requires Eigen."
#endif

// Follows the notation in:
// https://arxiv.org/pdf/1604.00772.pdf
// This is a (hopefully) faithful reproduction of the pseudocode in the arxiv review
// by Nikolaus Hansen.
// Comments referring to equations all refer to this arxiv review.
// A slide deck by the same author is given here:
// http://www.cmap.polytechnique.fr/~nikolaus.hansen/CmaTutorialGecco2023-no-audio.pdf
// which is also a very useful reference.

#ifndef BOOST_MATH_DEBUG_CMA_ES
#define BOOST_MATH_DEBUG_CMA_ES 0
#endif

namespace boost::math::optimization {

template <typename ArgumentContainer> struct cma_es_parameters {
  using Real = typename ArgumentContainer::value_type;
  ArgumentContainer lower_bounds;
  ArgumentContainer upper_bounds;
  size_t max_generations = 1000;
  ArgumentContainer const *initial_guess = nullptr;
  // In the reference, population size = \lambda.
  // If the population size is zero, it is set to equation (48) of the reference
  // and rounded up to the nearest multiple of threads:
  size_t population_size = 0;
  unsigned threads = std::thread::hardware_concurrency();
  // In the reference, learning_rate = c_m:
  Real learning_rate = 1;
};

template <typename ArgumentContainer>
void validate_cma_es_parameters(cma_es_parameters<ArgumentContainer> &params) {
  using Real = typename ArgumentContainer::value_type;
  using std::isfinite;
  using std::isnan;
  using std::log;
  using std::ceil;
  using std::floor;

  std::ostringstream oss;
  detail::validate_bounds(params.lower_bounds, params.upper_bounds);
  if (params.initial_guess) {
    detail::validate_initial_guess(*params.initial_guess, params.lower_bounds, params.upper_bounds);
  }
  if (params.threads == 0) {
    oss << __FILE__ << ":" << __LINE__ << ":" << __func__;
    oss << ": There must be at least one thread.";
    throw std::invalid_argument(oss.str());
  }
  auto n = params.upper_bounds.size();
  // Equation 48 of the arxiv review:
  if (params.population_size == 0) {
    auto tmp = 4.0 + floor(3*log(n));
    // But round to the nearest multiple of the thread count:
    auto k = static_cast<size_t>(std::ceil(tmp/params.threads));
    params.population_size = k*params.threads;
  }
  if (params.learning_rate <= Real(0) || !std::isfinite(params.learning_rate)) {
    oss << __FILE__ << ":" << __LINE__ << ":" << __func__;
    oss << ": The learning rate must be > 0, but got " << params.learning_rate << ".";
    throw std::invalid_argument(oss.str());
  }
}

template <typename ArgumentContainer, class Func, class URBG>
ArgumentContainer cma_es(
    const Func cost_function,
    cma_es_parameters<ArgumentContainer> &params,
    URBG &gen,
    std::invoke_result_t<Func, ArgumentContainer> target_value = std::numeric_limits<std::invoke_result_t<Func, ArgumentContainer>>::quiet_NaN(),
    std::atomic<bool> *cancellation = nullptr,
    std::atomic<std::invoke_result_t<Func, ArgumentContainer>> *current_minimum_cost = nullptr,
    std::vector<std::pair<ArgumentContainer, std::invoke_result_t<Func, ArgumentContainer>>> *queries = nullptr)
 {
  using Real = typename ArgumentContainer::value_type;
  using ResultType = std::invoke_result_t<Func, ArgumentContainer>;
  using std::abs;
  using std::log;
  using std::pow;
  using std::min;
  using std::max;
  using std::sqrt;
  using std::isnan;
  using std::uniform_real_distribution;
  using boost::math::optimization::detail::multivariate_normal_distribution;
  validate_cma_es_parameters(params);
  // n = dimension of problem:
  const size_t n = params.lower_bounds.size();
  std::atomic<bool> target_attained = false;
  std::atomic<ResultType> lowest_cost = std::numeric_limits<ResultType>::infinity();
  ArgumentContainer best_vector;
  // p_{c} := evolution path, equation (24) of the arxiv review:
  ArgumentContainer p_c;
  // p_{\sigma} := conjugate evolution path, equation (31) of the arxiv review:
  ArgumentContainer p_sigma;
  // Sadly necessary for code reuse:
  ArgumentContainer zero_vector;
  if constexpr (detail::has_resize_v<ArgumentContainer>) {
    best_vector.resize(n, std::numeric_limits<Real>::quiet_NaN());
    p_c.resize(n);
    p_sigma.resize(n);
    zero_vector.resize(n);
  }
  for (size_t i = 0; i < p_c.size(); ++i) {
    p_c[i] = Real(0);
    p_sigma[i] = Real(0);
    zero_vector[i] = Real(0);
  }
  // Table 1, \mu = floor(\lambda/2):
  size_t mu = params.population_size/2;
  std::vector<Real> w_prime(params.population_size, std::numeric_limits<Real>::quiet_NaN());
  for (size_t i = 0; i < params.population_size; ++i) {
    // Equation (49), but 0-indexed:
    w_prime[i] = log(static_cast<Real>(params.population_size + 1)/(2*(i+1)));
  }
  // Table 1, notes at top:
  Real positive_weight_sum = 0;
  Real sq_weight_sum = 0;
  for (size_t i = 0; i < mu; ++i) {
    assert(w_prime[i] > 0);
    positive_weight_sum += w_prime[i];
    sq_weight_sum += w_prime[i]*w_prime[i];
  }
  Real mu_eff = positive_weight_sum*positive_weight_sum/sq_weight_sum;
  assert(1 <= mu_eff);
  assert(mu_eff <= mu);
  Real negative_weight_sum = 0;
  sq_weight_sum = 0;
  for (size_t i = mu; i < params.population_size; ++i) {
    assert(w_prime[i] <= 0);
    negative_weight_sum += w_prime[i];
    sq_weight_sum += w_prime[i]*w_prime[i];
  }
  Real mu_eff_m = negative_weight_sum*negative_weight_sum/sq_weight_sum;
  // Equation (54):
  Real c_m = params.learning_rate;
  // Equation (55):
  Real c_sigma = (mu_eff + 2)/(n + mu_eff + 5);
  assert(c_sigma < 1);
  Real d_sigma = 1 + 2*max(Real(0), sqrt((mu_eff - 1)/(n + 1)) - 1) + c_sigma;
  // Equation (56):
  Real c_c = (4 + mu_eff/n)/(n + 4 + 2*mu_eff/n);
  assert(c_c <= 1);
  // Equation (57):
  Real c_1 = Real(2)/(pow(n + 1.3, 2) + mu_eff);
  // Equation (58)
  Real c_mu = min(1 - c_1, 2*(Real(0.25)  + mu_eff  + 1/mu_eff - 2)/((n+2)*(n+2) + mu_eff));
  assert(c_1 + c_mu <= Real(1));
  // Equation (50):
  Real alpha_mu_m = 1 + c_1/c_mu;
  // Equation (51):
  Real alpha_mu_eff_m = 1 + 2*mu_eff_m/(mu_eff + 2);
  // Equation (52):
  Real alpha_m_pos_def = (1- c_1 - c_mu)/(n*c_mu);
  // Equation (53):
  std::vector<Real> weights(params.population_size, std::numeric_limits<Real>::quiet_NaN());
  for (size_t i = 0; i < mu; ++i) {
    weights[i] = w_prime[i]/positive_weight_sum;
  }
  Real min_alpha = min(alpha_mu_m, min(alpha_mu_eff_m, alpha_m_pos_def));
  for (size_t i = mu; i < params.population_size; ++i) {
    weights[i] = min_alpha*w_prime[i]/abs(negative_weight_sum);
  }
  // mu:= number of parents, lambda := number of offspring.
  auto C = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>::Identity(n, n);
  ArgumentContainer mean_vector;
  Real sigma = 1;
  if (params.initial_guess) {
    mean_vector = *params.initial_guess;
  }
  else {
    // See the footnote in Table 1 of the arxiv review:
    sigma = 0.3*(params.upper_bounds[0] - params.lower_bounds[0]);
    mean_vector = detail::random_initial_population(params.lower_bounds, params.upper_bounds, 1, gen)[0];
  }
  auto initial_cost = cost_function(mean_vector);
  if (!isnan(initial_cost)) {
    best_vector = mean_vector;
    lowest_cost = initial_cost;
    if (current_minimum_cost) {
      *current_minimum_cost = initial_cost;
    }
  }

#if BOOST_MATH_DEBUG_CMA_ES
  {
    std::cout << __FILE__ << ":" << __LINE__ << ":" << __func__ << "\n";
    std::cout << "\tRunning a (" << params.population_size/2 << "/" << params.population_size/2 << "_W, " << params.population_size << ")-aCMA Evolutionary Strategy on " << params.threads << " threads.\n";
    std::cout << "\tInitial mean vector: {";
    for (size_t i = 0; i < n - 1; ++i) {
      std::cout << mean_vector[i] << ", ";
    }
    std::cout << mean_vector[n - 1] << "}.\n";
    std::cout << "\tCost: " << lowest_cost << ".\n";
    std::cout << "\tInitial step length: " << sigma << ".\n";
    std::cout << "\tVariance effective selection mass: " << mu_eff << ".\n";
    std::cout << "\tLearning rate for rank-one update of covariance matrix: " << c_1 << ".\n";
    std::cout << "\tLearning rate for rank-mu update of covariance matrix: " << c_mu << ".\n";
    std::cout << "\tDecay rate for cumulation path for step-size control: " << c_sigma << ".\n";
    std::cout << "\tLearning rate for the mean: " << c_m << ".\n";
    std::cout << "\tDamping parameter for step-size update: " << d_sigma << ".\n";
  }
#endif
  size_t generation = 0;

  std::vector<ArgumentContainer> ys(params.population_size);
  std::vector<ArgumentContainer> xs(params.population_size);
  std::vector<ResultType> costs(params.population_size, std::numeric_limits<ResultType>::quiet_NaN());
  if constexpr (detail::has_resize_v<ArgumentContainer>) {
    for (auto & x : xs) {
      x.resize(n, std::numeric_limits<Real>::quiet_NaN());
    }
    for (auto & y : ys) {
      y.resize(n, std::numeric_limits<Real>::quiet_NaN());
    }
  }
  do {
    auto mvnd = multivariate_normal_distribution<ArgumentContainer>(zero_vector, C);
    for (size_t k = 0; k < params.population_size; ++k) {
      auto & y = ys[k];
      auto & x = xs[k];
      mvnd(ys[k], gen); // equation (39) of figure 6
      for (size_t i = 0; i < n; ++i) {
        x[i] = mean_vector[i] + sigma*y[i]; // equation (40) of Figure 6.
      }
      costs[k] = cost_function(x);
      if (isnan(costs[k])) {
        continue;
      }
      if (costs[k] < lowest_cost) {
        lowest_cost = costs[k];
        if (current_minimum_cost) {
          //*current_minimum_cost = lowest_cost;
        }
        best_vector = x;
        if (lowest_cost < target_value) {
          break;
        }
#if BOOST_MATH_DEBUG_CMA_ES
        {
          std::cout << __FILE__ << ":" << __LINE__ << ":" << __func__ << "\n";
          std::cout << "\tNew lowest cost found: " << lowest_cost << "\n";
          std::cout << "\tBest vector: {";
          for (size_t i = 0; i < n - 1; ++i) {
            std::cout << best_vector[i] << ", ";
          }
          std::cout << best_vector[n - 1] << "}.\n";
        }
#endif
      }
      auto indices = detail::best_indices(costs);
      // Equation (41):

    }
  } while (generation++ < params.max_generations);

  return best_vector;
}

} // namespace boost::math::optimization
#endif
