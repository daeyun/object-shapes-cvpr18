//
// Created by daeyun on 6/30/17.
//

#pragma once

#include "common.h"

namespace mvshape {
namespace MP {
vector<pair<int, int>> SplitRange(int end, int n);;

static int N = 1;
static int process_num = 0;

int MultiFork(int work_size, int num_processes, pair<int, int> *start_end_indices);

void Join();

}
}
