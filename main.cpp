//
//  main.cpp
//  LR-x
//
//  Created by zhuangqh on 2017/6/16.
//  Copyright © 2017年 zhuangqh. All rights reserved.
//

#include <iostream>
#include <Eigen/Dense>
#include "io.h"
#include "LR.hpp"

using Eigen::MatrixXd;

int main() {

//  auto mat = LR::IO::load_txt("./train.txt", 1866819, 201);

  auto train = LR::IO::load_csv("../spamtrain.csv", 2760, 57);

  LR::LogisticRegression lr(1e-5, 10000, 0.2);

  lr.fit(train);

//  auto test = LR::IO::load_csv("../spamtest.csv", 1841, 57);
//
//  auto predict_tag = lr.predict(test.first);
//
//  std::cout << predict_tag << std::endl;
}
