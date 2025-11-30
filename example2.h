#pragma once
#include <iostream>
#include <vector>
#include <math.h>
#include <random>
#include <chrono>
#include <cassert>
#include <torch/torch.h>

#define WITHOUT_NUMPY
#include "matplotlibcpp/matplotlibcpp.h"

/*
	f(x) = (exp(abs(x_2 - 3)) - 30)/(x_1^2 + x_3^2 + 2x_4^2 + 4)
	Constraints:
	g_1(x) = (x_1 + x_3)^3 + 2x_4^2 - 10 <= 0
	g_2(x) = (x_2 - 1)^2 - 1 <= 0
	g_3(x) = 2x_1 + 4x_2 + x_3 + 1 <= 0
	g_4(x) = -1 - 2x_1 - 4x_2 - x_3 <= 0
*/

using namespace std;
namespace example2 {
	double f(const vector<double>& x) {
		return (exp(abs(x[1] - 3.0)) - 30.0) / (x[0] * x[0] + x[2] * x[2] + 2.0 * x[3] * x[3] + 4.0);
	}

	double g1(const vector<double>& x) {
		return pow(x[0] + x[2], 3.0) + 2.0 * x[3] * x[3] - 10.0;
	}

	double g2(const vector<double>& x) {
		return pow(x[1] - 1.0, 2.0) - 1.0;
	}

	double g3(const vector<double>& x) {
		return 2.0 * x[0] + 4.0 * x[1] + x[2] + 1.0;
	}

	vector<double> grad_f(const vector<double>& x) {
		vector<double> res(4, 0.0);
		double diff = x[1] - 3.0;
		double absdiff = abs(diff);
		double N = exp(absdiff) - 30.0;
		double D = x[0] * x[0] + x[2] * x[2] + 2.0 * x[3] * x[3] + 4.0;
		double D2 = D * D;

		res[0] = -2.0 * x[0] * N / D2;

		double sign = 0.0;
		if (absdiff > 1e-12)
			sign = diff / absdiff;
		res[1] = exp(absdiff) * sign / D;

		res[2] = -2.0 * x[2] * N / D2;
		res[3] = -4.0 * x[3] * N / D2;

		return res;
	}

	vector<double> grad_g1(const vector<double>& x) {
		vector<double> res(4, 0.0);
		double t = x[0] + x[2];
		double t2 = t * t;
		res[0] = 3.0 * t2;
		res[2] = 3.0 * t2;
		res[3] = 4.0 * x[3];
		return res;
	}

	vector<double> grad_g2(const vector<double>& x) {
		vector<double> res(4, 0.0);
		res[1] = 2.0 * (x[1] - 1.0);
		return res;
	}

	vector<double> grad_g3(const vector<double>& x) {
		vector<double> res = { 2.0, 4.0, 1.0, 0.0 };
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
		double res = 0.0;
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

	vector<double> projection(const vector<double>& y, int n, mt19937& rng, int max_iters = 100000, double lr = 0.001, double rho = 5.0) {
		(void)n;
		(void)rng;

		vector<double> x = y;

		for (int it = 0; it < max_iters; ++it) {
			vector<double> gradJ = x - y;
			double g1_val = g1(x);
			double g2_val = g2(x);
			double g3_val = g3(x);

			vector<double> g1_g = grad_g1(x);
			vector<double> g2_g = grad_g2(x);
			vector<double> g3_g = grad_g3(x);

			gradJ = gradJ + rho * max(g1_val, 0.0) * g1_g;
			gradJ = gradJ + rho * max(g2_val, 0.0) * g2_g;
			gradJ = gradJ + rho * g3_val * g3_g;

			x = x - lr * gradJ;
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

		double lda = 0.1;
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

			vector<double> x1(max_iters + 1), x2(max_iters + 1), x3(max_iters + 1), x4(max_iters + 1);

			for (int k = 0; k <= max_iters; ++k) {
				x1[k] = tr[k][0];
				x2[k] = tr[k][1];
				x3[k] = tr[k][2];
				x4[k] = tr[k][3];
			}


			if (i == 0) {
				plt::named_plot("$x_{1}(t)$", t, x1, "r-");
				plt::named_plot("$x_{2}(t)$", t, x2, "g-");
				plt::named_plot("$x_{3}(t)$", t, x3, "b-");
				plt::named_plot("$x_{4}(t)$", t, x4, "y-");
			}
			else {
				plt::plot(t, x1, "r-");
				plt::plot(t, x2, "g-");
				plt::plot(t, x3, "b-");
				plt::plot(t, x4, "y-");
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

		int n = 4; //dimension
		double epsilon = 0.1;
		(void)epsilon;
		double mu0 = unif(rng);
		double alpha = unif(rng);

		sol_all.reserve(num);
		val_all.reserve(num);

		for (int i = 0; i < num; ++i) {
			vector<double> x0 = { unif(rng), unif(rng), unif(rng), unif(rng)};
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