//
//  main.cpp
//  LR-x
//
//  Created by zhuangqh on 2017/6/16.
//  Copyright © 2017年 zhuangqh. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <omp.h>
#include "io.h"
#include "LR.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

void timer_wrapper(std::function<void()> func) {
  double startTime = omp_get_wtime();
  func();
  double stopTime = omp_get_wtime();

  std::cout << stopTime - startTime << std::endl;
}

void save_res(const char *filename, VectorXd &res) {
  std::ofstream f(filename);
  f << res;
  f.close();
}

int main() {

//  auto mat = LR::IO::load_txt("./train.txt", 1866819, 201);

  auto train = LR::IO::load_csv("../spamtrain.csv", 2760, 57);

  auto test = LR::IO::load_csv("../spamtest.csv", 1841, 57);

  LR::LogisticRegression lr(1e-5, 1000, 0);

  VectorXd res;

  timer_wrapper([&]() {
    lr.fit_naive(train);
  });
  res = lr.predict(test.first);
  save_res("naive.txt", res);

  timer_wrapper([&]() {
    lr.fit_vec(train);
  });
  res = lr.predict(test.first);
  save_res("vectorization.txt", res);

  timer_wrapper([&]() {
    lr.fit_parallel(train);
  });
  res = lr.predict(test.first);
  save_res("parallel.txt", res);

  return 0;
}
