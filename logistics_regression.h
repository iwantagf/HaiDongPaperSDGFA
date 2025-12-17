#include <iostream>
#include <torch/torch.h>
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <math.h>
#include <chrono>
#include <random>


using namespace std;
using torch::Tensor;


class DataFrame {
public:
	vector<string> columns;
	map<string, size_t> cols_id;

	vector<vector<double>> data;

	Tensor X;

	bool load_csv(const string& file_path) {
		ifstream fin(file_path);

		if (!fin.is_open()) {
			return false;
		}

		string line;
		bool is_first_line = true;

		vector<double> flat;
		size_t ncols_expected = 0;

		data.clear();
		columns.clear();
		cols_id.clear();
		X = Tensor(); // reset

		while (getline(fin, line)) {
			if (line.empty())
				continue;

			stringstream ss(line);
			string cell;
			vector<string> cells;

			while (getline(ss, cell, ',')) {
				cells.push_back(cell);
			}

			if (is_first_line) {
				is_first_line = false;
				columns = cells;

				for (size_t i = 0; i < columns.size(); ++i)
					cols_id[columns[i]] = i;

				ncols_expected = columns.size();
			}
			else {
				if (cells.size() != ncols_expected) {
					return false;
				}

				vector<double> row;
				row.reserve(cells.size());

				for (string& s : cells) {
					double v = stod(s);
					row.push_back(v);
					flat.push_back(v);
				}

				data.push_back(move(row));
			}
		}

		if (!data.empty() && !columns.empty()) {
			const int64_t nrows = (int64_t)data.size();
			const int64_t ncols = (int64_t)columns.size();
			X = torch::from_blob(flat.data(), { nrows, ncols }, torch::kDouble).clone();
		}

		return true;
	}

	size_t nrows() const {
		return data.size();
	}

	size_t ncols() const {
		return columns.size();
	}

	double& at(size_t r, const string& col_name) {
		return data[r][cols_id.at(col_name)];
	}

	double at(size_t r, const string& col_name) const {
		return data[r][cols_id.at(col_name)];
	}

	vector<double> get_col(const string& col_name) const {
		size_t j = cols_id.at(col_name);
		vector<double> res;
		res.reserve(nrows());

		for (size_t i = 0; i < nrows(); ++i)
			res.push_back(data[i][j]);

		return res;
	}


	void normalize_col(const string& col_name) {
		size_t j = cols_id.at(col_name);

		if (X.defined()) {
			auto col = X.index({ torch::indexing::Slice(), (int64_t)j });
			auto mean = col.mean();
			auto var = (col - mean).pow(2).mean();
			auto de = (var + 1e-12).sqrt();

			auto new_col = (col - mean) / de;
			X.index_put_({ torch::indexing::Slice(), (int64_t)j }, new_col);


			auto new_col_cpu = new_col.contiguous().to(torch::kCPU);
			auto ptr = new_col_cpu.data_ptr<double>();
			for (size_t i = 0; i < nrows(); ++i) {
				data[i][j] = ptr[i];
			}

			return;
		}

		vector<double> col = get_col(col_name);

		double mean = 0, var = 0;

		for (double v : col)
			mean += v;

		mean /= col.size();

		for (double v : col)
			var += (v - mean) * (v - mean);

		var /= col.size();

		double de = sqrt(var + 1e-12);

		for (size_t i = 0; i < nrows(); ++i)
			data[i][j] = (data[i][j] - mean) / de;

		return;
	}
};

namespace regression {
	torch::Device device(torch::kCPU);

	void train_test_split(Tensor data, Tensor& train, Tensor& test) {
		// data: (N, D)
		double test_ratio = 0.36;
		int64_t seed = 42;

		if (!data.defined() || data.dim() != 2) {
			train = Tensor();
			test = Tensor();
			return;
		}

		const int64_t N = data.size(0);
		int64_t n_test = (int64_t)llround(test_ratio * (double)N);

		if (n_test <= 0) n_test = 1;
		if (n_test >= N) n_test = N - 1;

		torch::manual_seed(seed);
		auto perm = torch::randperm(N, torch::TensorOptions().dtype(torch::kLong).device(data.device()));

		auto test_idx = perm.index({ torch::indexing::Slice(0, n_test) });
		auto train_idx = perm.index({ torch::indexing::Slice(n_test, torch::indexing::None) });

		test = data.index_select(0, test_idx).contiguous();
		train = data.index_select(0, train_idx).contiguous();
	}

	Tensor bce_logits_full(const Tensor& X, const Tensor& y, const Tensor& w, const Tensor& b, double l2 = 0.0) {
		auto logits = torch::matmul(X, w) + b;

		auto loss = torch::nn::functional::binary_cross_entropy_with_logits(
			logits,
			y,
			torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions().reduction(torch::kMean)
		);

		if (l2 > 0.0)
			loss = loss + 0.5 * l2 * (w * w).sum();

		return loss;
	}

	double accuracy(const Tensor& X, const Tensor& y, const Tensor& w, const Tensor& b) {
		auto logits = torch::matmul(X, w) + b;
		auto pred = (torch::sigmoid(logits) >= 0.5).to(y.scalar_type());

		return (pred == y).to(torch::kDouble).mean().item<double>();
	}

	void print_confusion_matrix(int TP, int FP, int TN, int FN) {
		cout << "\n=== Confusion Matrix ===\n";
		cout << "                       Predicted\n";
		cout << "                     |   0   |   1  |\n";
		cout << "---------------------------------------\n";
		cout << "Actual |     0       |  " << setw(5) << TN << " | " << setw(5) << FP << " |\n";
		cout << "       |     1       |  " << setw(5) << FN << " | " << setw(5) << TP << " |\n";
		cout << "---------------------------------------\n\n";
	}

	double f_measure(const Tensor& X, const Tensor& y, const Tensor& w, const Tensor& b) {
		auto logits = torch::matmul(X, w) + b;
		auto pred = (torch::sigmoid(logits) >= 0.5).to(torch::kInt32);
		auto yt = y.to(torch::kInt32);

		auto tp = ((pred == 1) & (yt == 1)).sum().item<int>();
		auto fp = ((pred == 1) & (yt == 0)).sum().item<int>();
		auto fn = ((pred == 0) & (yt == 1)).sum().item<int>();
		auto tn = ((pred == 0) & (yt == 0)).sum().item<int>();

		print_confusion_matrix(tp, fp, tn, fn);
		double precision = double(tp) / (tp + fp + 1e-12);
		double recall = double(tp) / (tp + fn + 1e-12);

		return 2.0 * precision * recall / (precision + recall + 1e-12);
	}

	void logistic_regression(Tensor& X_train, Tensor& y_train, Tensor& X_test, Tensor& y_test) {
		int epochs = 200;
		int batch_size = 256;
		double ld0 = 36;
		double kappa = 0.936;
		double sigma = 0.369;
		double l2 = 1e-2;
		double max_tries = 10;

		X_train = X_train.contiguous().to(device).to(torch::kFloat);
		y_train = y_train.contiguous().to(device).to(torch::kFloat);
		X_test = X_test.contiguous().to(device).to(torch::kFloat);
		y_test = y_test.contiguous().to(device).to(torch::kFloat);

		const long long D = X_train.size(1);

		Tensor w = torch::zeros({ D }, torch::TensorOptions().dtype(torch::kFloat).device(device).requires_grad(true));
		Tensor b = torch::zeros({ 1 }, torch::TensorOptions().dtype(torch::kFloat).device(device).requires_grad(true));

		double lda = ld0;


		for (int iter = 0; iter < epochs; ++iter) {
			auto loss = bce_logits_full(X_train, y_train, w, b, l2);
			auto grads = torch::autograd::grad({ loss }, { w, b });

			auto gw = grads[0].detach();
			auto gb = grads[1].detach();


			double gnorm2 = ((gw * gw).sum() + (gb * gb).sum()).item<double>();

			double lda_try = lda;
			bool ok = false;

			for (int trying = 0; trying < max_tries; ++trying) {
				auto w_try = w.detach() - (float)lda_try * gw;
				auto b_try = b.detach() - (float)lda_try * gb;

				auto loss_try = bce_logits_full(X_train, y_train, w_try, b_try, l2);

				double tmp = (loss_try - loss).item<double>() + sigma * lda_try * gnorm2;

				if (tmp <= 0.0) {
					w = w_try.clone().detach().set_requires_grad(true);
					b = b_try.clone().detach().set_requires_grad(true);
					lda = lda_try;
					ok = true;
					break;
				}

				lda_try *= kappa;
			}

			if (!ok)
				lda *= kappa;


			if (iter % 5 == 0) {
				auto train_loss = bce_logits_full(X_train, y_train, w, b, l2).item<double>();
				auto test_loss = bce_logits_full(X_test, y_test, w, b, l2).item<double>();
				double train_acc = accuracy(X_train, y_train, w, b);
				double test_acc = accuracy(X_test, y_test, w, b);

				cout << "[Iter " << iter << "] lambda=" << lda
					<< " train_loss=" << train_loss
					<< " test_loss=" << test_loss
					<< " train_acc=" << train_acc
					<< " test_acc=" << test_acc
					<< "\n";
			}

		}

		cout << "Train:\n";
		double train_f1 = f_measure(X_train, y_train, w, b);

		cout << "Test:\n";
		double test_f1 = f_measure(X_test, y_test, w, b);

		cout << "[Final] "
			<< " train_f1=" << train_f1
			<< " test_f1=" << test_f1
			<< "\n";
	}

	void solve() {
		if (torch::cuda::is_available()) {
			device = torch::Device(torch::kCUDA);
			cout << "CUDA is available, using GPU to train!\n";
		}
		else
			cout << "CUDA is not avaiable, using CPU to train\n";



		mt19937 rng(42);
		const string df_path = "C:\\Users\\PC\\Desktop\\Personal Project\\SGDA\\archive\\creditcard_2023.csv";

		DataFrame df;

		cout << df.load_csv(df_path) << '\n';
		cout << df.nrows() << " " << df.ncols() << '\n';

		for (size_t j = 0; j + 1 < df.ncols(); ++j) {
			df.normalize_col(df.columns[j]);
		}

		Tensor train_t, test_t;
		train_test_split(df.X, train_t, test_t);

		cout << "tensor split: " << train_t.sizes() << " " << test_t.sizes() << '\n';

		auto X_train = train_t.index({ torch::indexing::Slice(), torch::indexing::Slice(0, -1) });
		auto y_train = train_t.index({ torch::indexing::Slice(), -1 });
		auto X_test = test_t.index({ torch::indexing::Slice(), torch::indexing::Slice(0, -1) });
		auto y_test = test_t.index({ torch::indexing::Slice(), -1 });

		cout << "X_train: " << X_train.sizes() << '\n';
		cout << "y_train: " << y_train.sizes() << '\n';


		logistic_regression(X_train, y_train, X_test, y_test);
	}
}