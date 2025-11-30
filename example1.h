#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <cassert>

#define WITHOUT_NUMPY
#include "matplotlibcpp/matplotlibcpp.h"

/*
f(x) = (x1^2 + x2^2 + 3)/(1 + 2x1 + 8x2)
C = {x = (x1, x2)^T in R^2 | g_1(x) = -x1^2 - 2x1x2 <= -4, g_2(x) = x1 >= 0, g_3(x) = x2 >= 0}
*/

using namespace std;
namespace example1 {
	double f(vector<double> x) {
		return (x[0] * x[0] + x[1] * x[1] + 3) / (1 + 2 * x[0] + 8 * x[1]);
	}

	vector<double> grad_f(vector<double> x) {
		vector<double> res(2, 0.0);

		double x1 = x[0];
		double x2 = x[1];
		//df/dx1 = (2x1 * (1 + 2x1 + 8x2) - 2 * (x1^2 + x2^2 + 3))/(1 + 2x1 + 8x2)^2
		double denominator = (1 + 2 * x1 + 8 * x2) * (1 + 2 * x1 + 8 * x2);

		res[0] = (2 * x1 * (1 + 2 * x1 + 8 * x2) - 2 * (x1 * x1 + x2 * x2 + 3)) / denominator;
		res[1] = (2 * x2 * (1 + 2 * x1 + 8 * x2) - 8 * (x1 * x1 + x2 * x2 + 3)) / denominator;

		return res;
	}

	double g1(vector<double> x) {
		return -x[0] * x[0] - 2 * x[1] * x[0] + 4;
	}

	double g2(vector<double> x) {
		return -x[0];
	}

	double g3(vector<double> x) {
		return -x[1];
	}

	vector<double> grad_g1(vector<double> x) {
		vector<double> res(2, 0.0);

		res[0] = -2 * x[0] - 2 * x[1];
		res[1] = -2 * x[0];

		return res;
	}

	template<typename T>
	vector<T> operator - (const vector<T>& a, const vector<T>& b) {
		assert(a.size() == b.size());
		vector<T> res(a.size());
		for (size_t i = 0; i < a.size(); ++i)
			res[i] = a[i] - b[i];
		return res;
	}

	template<typename T>
	vector<T> operator + (const vector<T>& a, const vector<T>& b) {
		assert(a.size() == b.size());
		vector<T> res(a.size());
		for (size_t i = 0; i < a.size(); ++i)
			res[i] = a[i] + b[i];
		return res;
	}

	template<typename T>
	double dot(const vector<T>& a, const vector<T>& b) {
		assert(a.size() == b.size());
		double res = 0;
		for (size_t i = 0; i < a.size(); ++i)
			res += a[i] * b[i];
		return res;
	}

	template<typename T>
	vector<T> operator * (const double& a, const vector<T>& b) {
		vector<T> res(b.size());
		for (size_t i = 0; i < b.size(); ++i)
			res[i] = a * b[i];
		return res;
	}

	/*
		P(X) = argmin ||x - y||, y in C
		find argmin 1/2 ||x - y||^2 + rho * max(0, g1(x))
		constraint: x1, x2 <= 0, g1(x) <= 0
	*/

	vector<double> projection(const vector<double>& y, int n, mt19937& rng, int max_iters = 100000, double lr = 0.001, double rho = 100.0) {
		(void)n;

		uniform_real_distribution<double> unif(0.0, 1.0);
		vector<double> x0(2, 0.0);

		x0[0] = unif(rng);
		x0[1] = unif(rng);

		vector<double> x = x0;

		for (int it = 0; it < max_iters; ++it) {
			vector<double> gradJ = x - y;
			double g1_val = g1(x);

			if (g1_val > 0.0) {
				vector<double> g1_g = grad_g1(x);
				gradJ = gradJ + 2.0 * rho * g1_val * g1_g;
			}

			x = x - lr * gradJ;

			x[0] = max(0.0, x[0]);
			x[1] = max(0.0, x[1]);
		}

		return x;
	}

	struct RunResult {
		vector< vector<double> > res;
		vector<double> val;
	};

	RunResult run_nonsmooth(const vector<double>& X, int max_iters, int n, double alpha, double mu0, mt19937& rng) {
		(void)n;
		(void)alpha;
		double mut = mu0;
		(void)mut;

		RunResult R;
		R.res.reserve(max_iters + 1);
		R.val.reserve(max_iters + 1);

		double lda = 1;
		double sigma = 0.1;

		uniform_real_distribution<double> unif(0.0, 1.0);
		double K = unif(rng);

		R.res.push_back(X);
		R.val.push_back(f(X));

		vector<double> x = X;
		vector<double> x_pre = X;

		for (int i = 1; i <= max_iters; ++i) {
			vector<double> y = x - lda * grad_f(x);
			x_pre = x;
			x = projection(y, n, rng);
			if (f(x) - f(x_pre) + sigma * dot(grad_f(x_pre), x_pre - x) <= 0) {
				lda = lda;
			}
			else
				lda = K * lda;

			R.res.push_back(x);
			R.val.push_back(f(x));
		}

		return R;
	}

	namespace plt = matplotlibcpp;

	void plot_x(const vector< vector< vector<double> > >& sol_all, int count, int max_iters) {
		plt::figure_size(800, 800);
		plt::figure();

		vector<int> t(max_iters + 1);
		for (int i = 0; i <= max_iters; ++i)
			t[i] = i;

		for (int i = 0; i < count; ++i) {
			const auto& tr = sol_all[i];

			vector<double> x1(max_iters + 1), x2(max_iters + 1);

			for (int k = 0; k <= max_iters; ++k) {
				x1[k] = tr[k][0];
				x2[k] = tr[k][1];
			}


			if (i == 0) {
				plt::named_plot("$x_{1}(t)$", t, x1, "r-");
				plt::named_plot("$x_{2}(t)$", t, x2, "g-");
			}
			else {
				plt::plot(t, x1, "r-");
				plt::plot(t, x2, "g-");
			}
		}

		plt::xlabel("iteration");
		plt::ylabel("x(t)");
		map<string, string> leg_kw;
		leg_kw["loc"] = "best";
		plt::legend(leg_kw);
		plt::show();
	}

	void solve() {
		mt19937 rng(42);

		int num = 3;
		int max_iters = 100;
		//int max_iters1 = 100;
		int count = 0;
		vector< vector< vector<double> > > sol_all;
		vector< vector<double> > val_all;

		uniform_real_distribution<double> unif(0.0, 1.0);

		int n = 2; //dimension
		double epsilon = 0.1;
		(void)epsilon;
		double mu0 = unif(rng);
		double alpha = unif(rng);

		sol_all.reserve(num);
		val_all.reserve(num);

		for (int i = 0; i < num; ++i) {
			vector<double> x0 = { unif(rng), unif(rng) };
			x0 = projection(x0, n, rng);
			++count;

			auto t_start = chrono::high_resolution_clock::now();
			RunResult R = run_nonsmooth(x0, max_iters, n, alpha, mu0, rng);
			auto t_end = chrono::high_resolution_clock::now();
			double elapsed_time = chrono::duration<double>(t_end - t_start).count();

			cout << "GDA run " << i + 1 << " time: " << elapsed_time << "s, Value of f: " << R.val.back() << " \n";

			sol_all.push_back(R.res);
			val_all.push_back(R.val);
		}

		plot_x(sol_all, count, max_iters);
	}
}