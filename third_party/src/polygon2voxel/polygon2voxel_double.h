//
// Created by daeyun on 7/15/17.
//

#pragma once

#include <stdint.h>

void polygon2voxel(const int32_t *FacesA, const int32_t *FacesB, const int32_t *FacesC, const int32_t num_faces,
                   const double *VerticesX, const double *VerticesY, const double *VerticesZ,
                   const int32_t* VolumeDims, int32_t wrap, uint8_t* OutVolume);
