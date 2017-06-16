//
//  LR.cpp
//  LR-x
//
//  Created by zhuangqh on 2017/6/16.
//  Copyright © 2017年 zhuangqh. All rights reserved.
//

#include <iostream>
#include "LR.hpp"

namespace LR {

  double g(double x) {
    return 1.0 / (1.0 + std::exp(-x));
  }

  // Vectorization
  void LogisticRegression::fit_vec(std::pair<Eigen::MatrixXd, Eigen::VectorXd> train) {
    m = train.first.rows();
    n = train.first.cols();

    theta.setZero(n);
    probas.setZero(m);

    for (size_t i = 0; i < maxIter; i++) {
      // A = x * theta
      VectorXd inner = train.first * theta;

      // E = g(A)
      this->probas = inner.unaryExpr(std::ptr_fun(g));

      // gw = 1/m * xT * (E - y)
      VectorXd gw = 1.0 / this->m * (train.first.transpose() * (this->probas - train.second));

      // theta -= alpha * gw
      this->theta -= this->alpha * gw;

      std::cout << gw.norm() << std::endl;
      // check for convergence
      if (this->epsilon > gw.norm()) {
        break;
      }
    }
  }

  VectorXd LogisticRegression::predict(MatrixXd test) {
    return (test * theta).unaryExpr(std::ptr_fun(g));
  }

}
