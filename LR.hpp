//
//  LR.hpp
//  LR-x
//
//  Created by zhuangqh on 2017/6/16.
//  Copyright © 2017年 zhuangqh. All rights reserved.
//

#pragma once

#include <Eigen/Core>

namespace LR {

  using Eigen::VectorXd;
  using Eigen::MatrixXd;

  class LogisticRegression {
  private:
    double alpha;
    double lambda;
    double epsilon;
    int maxIter;

    VectorXd theta;
    VectorXd probas;

    long m;
    long n;
  public:
    LogisticRegression(double alpha, int maxIter, double lambda, double epsilon=1e-5)
        : alpha(alpha), maxIter(maxIter), lambda(lambda), epsilon(epsilon) {}

    void fit_vec(std::pair<MatrixXd, VectorXd>);

    void fit_naive(std::pair<MatrixXd, VectorXd>);

    VectorXd predict(MatrixXd);
  };

}
