#include <torch/torch.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <random>

#define WITHOUT_NUMPY
#include "matplotlibcpp/matplotlibcpp.h"

using torch::Tensor;

class DataFrame {
public:
    std::vector<std::string> columns;
    std::map<std::string, size_t> cols_id;

    std::vector<std::vector<double>> data;
    Tensor X;

    bool load_csv(const std::string& file_path) {
        std::ifstream fin(file_path);
        if (!fin.is_open()) return false;

        std::string line;
        bool is_first_line = true;

        std::vector<double> flat;
        size_t ncols_expected = 0;

        data.clear();
        columns.clear();
        cols_id.clear();
        X = Tensor();

        while (std::getline(fin, line)) {
            if (line.empty()) continue;

            std::stringstream ss(line);
            std::string cell;
            std::vector<std::string> cells;

            while (std::getline(ss, cell, ',')) {
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
                if (cells.size() != ncols_expected) return false;

                std::vector<double> row;
                row.reserve(cells.size());

                for (std::string& s : cells) {
                    double v = std::stod(s);
                    row.push_back(v);
                    flat.push_back(v);
                }

                data.push_back(std::move(row));
            }
        }

        if (!data.empty() && !columns.empty()) {
            const int64_t nrows = (int64_t)data.size();
            const int64_t ncols = (int64_t)columns.size();
            X = torch::from_blob(flat.data(), { nrows, ncols }, torch::kDouble).clone();
        }

        return true;
    }

    size_t nrows() const { return data.size(); }
    size_t ncols() const { return columns.size(); }

    void normalize_col(const std::string& col_name) {
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
            for (size_t i = 0; i < nrows(); ++i) data[i][j] = ptr[i];
            return;
        }

        double mean = 0.0, var = 0.0;
        for (size_t i = 0; i < nrows(); ++i) mean += data[i][j];
        mean /= (double)nrows();

        for (size_t i = 0; i < nrows(); ++i) {
            double d = data[i][j] - mean;
            var += d * d;
        }
        var /= (double)nrows();

        double de = std::sqrt(var + 1e-12);
        for (size_t i = 0; i < nrows(); ++i)
            data[i][j] = (data[i][j] - mean) / de;
    }
};

namespace regression {

    namespace plt = matplotlibcpp;

    torch::Device device(torch::kCPU);

    struct History {
        std::vector<double> iters;
        std::vector<double> train_loss, test_loss;
        std::vector<double> train_acc, test_acc;
    };

    void train_test_split(Tensor data, Tensor& train, Tensor& test) {
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

        if (l2 > 0.0) loss = loss + 0.5 * l2 * (w * w).sum();
        return loss;
    }

    double accuracy(const Tensor& X, const Tensor& y, const Tensor& w, const Tensor& b) {
        auto logits = torch::matmul(X, w) + b;
        auto pred = (torch::sigmoid(logits) >= 0.5).to(y.scalar_type());
        return (pred == y).to(torch::kDouble).mean().item<double>();
    }

    void print_confusion_matrix(int TP, int FP, int TN, int FN) {
        std::cout << "\n=== Confusion Matrix ===\n";
        std::cout << "                       Predicted\n";
        std::cout << "                     |   0   |   1  |\n";
        std::cout << "---------------------------------------\n";
        std::cout << "Actual |     0       |  " << std::setw(5) << TN << " | " << std::setw(5) << FP << " |\n";
        std::cout << "       |     1       |  " << std::setw(5) << FN << " | " << std::setw(5) << TP << " |\n";
        std::cout << "---------------------------------------\n\n";
    }

    double f1_measure(const Tensor& X, const Tensor& y, const Tensor& w, const Tensor& b) {
        auto logits = torch::matmul(X, w) + b;
        auto pred = (torch::sigmoid(logits) >= 0.5).to(torch::kInt32);
        auto yt = y.to(torch::kInt32);

        int tp = ((pred == 1) & (yt == 1)).sum().item<int>();
        int fp = ((pred == 1) & (yt == 0)).sum().item<int>();
        int fn = ((pred == 0) & (yt == 1)).sum().item<int>();
        int tn = ((pred == 0) & (yt == 0)).sum().item<int>();

        print_confusion_matrix(tp, fp, tn, fn);

        double precision = double(tp) / (tp + fp + 1e-12);
        double recall = double(tp) / (tp + fn + 1e-12);
        return 2.0 * precision * recall / (precision + recall + 1e-12);
    }

    History logistic_regression_GDA(Tensor X_train, Tensor y_train, Tensor X_test, Tensor y_test) {
        int epochs = 200;

        double ld0 = 3.6;
        double kappa = 0.9;
        double sigma = 1e-4;
        double l2 = 1e-2;
        int   max_tries = 10;

        X_train = X_train.contiguous().to(device).to(torch::kFloat);
        y_train = y_train.contiguous().to(device).to(torch::kFloat);
        X_test = X_test.contiguous().to(device).to(torch::kFloat);
        y_test = y_test.contiguous().to(device).to(torch::kFloat);

        const int64_t D = X_train.size(1);

        Tensor w = torch::zeros({ D }, torch::TensorOptions().dtype(torch::kFloat).device(device).requires_grad(true));
        Tensor b = torch::zeros({ 1 }, torch::TensorOptions().dtype(torch::kFloat).device(device).requires_grad(true));

        double lda = ld0;

        History hist;
        const int log_every = 5;

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

            if (!ok) lda *= kappa;

            if (iter % log_every == 0) {
                double tr_loss = bce_logits_full(X_train, y_train, w, b, l2).item<double>();
                double te_loss = bce_logits_full(X_test, y_test, w, b, l2).item<double>();
                double tr_acc = accuracy(X_train, y_train, w, b);
                double te_acc = accuracy(X_test, y_test, w, b);

                hist.iters.push_back((double)iter);
                hist.train_loss.push_back(tr_loss);
                hist.test_loss.push_back(te_loss);
                hist.train_acc.push_back(tr_acc);
                hist.test_acc.push_back(te_acc);

                std::cout << "[GDA Iter " << iter << "] lambda=" << lda
                    << " train_loss=" << tr_loss
                    << " test_loss=" << te_loss
                    << " train_acc=" << tr_acc
                    << " test_acc=" << te_acc
                    << "\n";
            }
        }

        std::cout << "Line-search Train:\n";
        double train_f1 = f1_measure(X_train, y_train, w, b);

        std::cout << "Line-search Test:\n";
        double test_f1 = f1_measure(X_test, y_test, w, b);

        std::cout << "[GDA Final] train_f1=" << train_f1 << " test_f1=" << test_f1 << "\n";
        return hist;
    }


    History logistic_regression_gd(Tensor X_train, Tensor y_train, Tensor X_test, Tensor y_test,
        double lr = 0.01, int epochs = 200, double l2 = 1e-2) {
        X_train = X_train.contiguous().to(device).to(torch::kFloat);
        y_train = y_train.contiguous().to(device).to(torch::kFloat);
        X_test = X_test.contiguous().to(device).to(torch::kFloat);
        y_test = y_test.contiguous().to(device).to(torch::kFloat);

        const int64_t D = X_train.size(1);

        Tensor w = torch::zeros({ D }, torch::TensorOptions().dtype(torch::kFloat).device(device).requires_grad(true));
        Tensor b = torch::zeros({ 1 }, torch::TensorOptions().dtype(torch::kFloat).device(device).requires_grad(true));

        History hist;
        const int log_every = 1;

        for (int iter = 0; iter < epochs; ++iter) {
            auto loss = bce_logits_full(X_train, y_train, w, b, l2);
            auto grads = torch::autograd::grad({ loss }, { w, b });

            auto gw = grads[0];
            auto gb = grads[1];

            w = (w - (float)lr * gw).detach().set_requires_grad(true);
            b = (b - (float)lr * gb).detach().set_requires_grad(true);

            if (iter % log_every == 0) {
                double tr_loss = bce_logits_full(X_train, y_train, w, b, l2).item<double>();
                double te_loss = bce_logits_full(X_test, y_test, w, b, l2).item<double>();
                double tr_acc = accuracy(X_train, y_train, w, b);
                double te_acc = accuracy(X_test, y_test, w, b);

                hist.iters.push_back((double)iter);
                hist.train_loss.push_back(tr_loss);
                hist.test_loss.push_back(te_loss);
                hist.train_acc.push_back(tr_acc);
                hist.test_acc.push_back(te_acc);

                std::cout << "[GD Iter " << iter << "] lr=" << lr
                    << " train_loss=" << tr_loss
                    << " test_loss=" << te_loss
                    << " train_acc=" << tr_acc
                    << " test_acc=" << te_acc
                    << "\n";
            }
        }

        std::cout << "Plain GD Train:\n";
        double train_f1 = f1_measure(X_train, y_train, w, b);

        std::cout << "Plain GD Test:\n";
        double test_f1 = f1_measure(X_test, y_test, w, b);

        std::cout << "[GD Final] train_f1=" << train_f1 << " test_f1=" << test_f1 << "\n";
        return hist;
    }

    // ============================
    // PLOT COMPARISON
    // ============================
    void plot_comparison(const History& gd, const History& ls) {
        // LOSS
        plt::figure();
        plt::named_plot("GD", gd.iters, gd.test_loss);
        plt::named_plot("GDA", ls.iters, ls.test_loss);
        plt::title("Loss comparison");
        plt::xlabel("Iteration");
        plt::ylabel("BCEWithLogits + L2");
        plt::legend();
        plt::grid(true);

        // ACC
        plt::figure();
        plt::named_plot("GD", gd.iters, gd.test_acc);
        plt::named_plot("GDA", ls.iters, ls.test_acc);
        plt::title("Accuracy comparison");
        plt::xlabel("Iteration");
        plt::ylabel("Accuracy");
        plt::legend();
        plt::grid(true);

        plt::show();
    }

    void solve() {
        if (torch::cuda::is_available()) {
            device = torch::Device(torch::kCUDA);
            std::cout << "CUDA is available, using GPU to train!\n";
        }
        else {
            std::cout << "CUDA is not available, using CPU to train\n";
        }

        const std::string df_path = "C:\\Users\\PC\\Desktop\\Personal Project\\SGDA\\archive\\creditcard_2023.csv";

        DataFrame df;
        std::cout << df.load_csv(df_path) << "\n";
        std::cout << df.nrows() << " " << df.ncols() << "\n";

        // normalize all feature columns except last (label)
        for (size_t j = 0; j + 1 < df.ncols(); ++j) {
            df.normalize_col(df.columns[j]);
        }

        Tensor train_t, test_t;
        train_test_split(df.X, train_t, test_t);
        std::cout << "tensor split: " << train_t.sizes() << " " << test_t.sizes() << "\n";

        auto X_train = train_t.index({ torch::indexing::Slice(), torch::indexing::Slice(0, -1) });
        auto y_train = train_t.index({ torch::indexing::Slice(), -1 });
        auto X_test = test_t.index({ torch::indexing::Slice(), torch::indexing::Slice(0, -1) });
        auto y_test = test_t.index({ torch::indexing::Slice(), -1 });

        std::cout << "X_train: " << X_train.sizes() << "\n";
        std::cout << "y_train: " << y_train.sizes() << "\n";

        // Run BOTH methods
        auto hist_gda = logistic_regression_GDA(X_train, y_train, X_test, y_test);
        auto hist_gd = logistic_regression_gd(X_train, y_train, X_test, y_test, 0.01, 200, 1e-2);

        // Compare plots
        plot_comparison(hist_gd, hist_gda);
    }

}