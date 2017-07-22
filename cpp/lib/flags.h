//
// Created by daeyun on 6/29/17.
//
#pragma once

#include <gflags/gflags_declare.h>

//namespace google {}
//namespace gflags=google;

DECLARE_string(data_dir);
DECLARE_string(out_dir);
DECLARE_string(resources_dir);
DECLARE_bool(is_test_mode);
DECLARE_string(tf_model);
DECLARE_string(run_id);
DECLARE_int32(batch_size);
DECLARE_string(default_device);
