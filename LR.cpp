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

  void LogisticRegression::fit(std::pair<Eigen::MatrixXd, Eigen::VectorXd> train) {
    m = train.first.rows();
    n = train.first.cols();

    theta.setZero(n);
    probas.setZero(m);

    for (size_t i = 0; i < maxIter; i++) {
      VectorXd inner = train.first * theta;

      this->probas = inner.unaryExpr(std::ptr_fun(g));

      VectorXd gw = 1.0 / this->m * (train.first.transpose() * (this->probas - train.second));

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
