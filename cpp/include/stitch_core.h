#pragma once

#include <stdint.h>

#ifdef _WIN32
#define SC_API __declspec(dllexport)
#else
#define SC_API
#endif

extern "C" {

typedef struct ScCanvas {
    double min_x;
    double min_y;
    double fit;
    int32_t width_before_rot;
    int32_t height_before_rot;
    int32_t out_width;
    int32_t out_height;
} ScCanvas;

/* Working-scale BGR image. Returns a handle >= 0, or -1. */
SC_API int sc_open(const char* utf8_path, double scale, char* err, int err_len);

SC_API void sc_close(int handle);

SC_API int sc_size(int handle, int* width, int* height);
SC_API int sc_full_size(int handle, int* width, int* height);

/* Row-major gray thumbnail, tw * th bytes. */
SC_API int sc_thumb(int handle, int tw, int th, uint8_t* out);

/* Circle centers in working-scale pixels, xy interleaved. Returns the count. */
SC_API int sc_grid(int handle, double* xy, int max_points);

/*
 * Homography mapping pixels of every image in A onto pixels of every image in B.
 * Matches are pooled, then fit with RANSAC. H9 is row-major 3x3.
 */
SC_API int sc_sift_homography(
    const int* handles_a,
    int count_a,
    const int* handles_b,
    int count_b,
    double* H9,
    int* matches,
    int* inliers,
    char* err,
    int err_len);

/* ransac_px <= 0 fits with every point. */
/* NCC of the drawing in a ring around dst_xy after warping src by H. */
SC_API int sc_annulus_ncc(
    int handle_src,
    int handle_dst,
    const double* H_src_to_dst,
    const double* dst_xy,
    int n,
    double* score);

SC_API int sc_fit_homography(
    const double* src_xy,
    const double* dst_xy,
    int n,
    double ransac_px,
    double* H9,
    int* inliers);

/* Same-camera frames, board masked out. Returns a new handle. */
SC_API int sc_composite(const int* handles, int count, char* err, int err_len);

SC_API int sc_save_overlap(
    int handle_src,
    int handle_dst,
    const double* H_src_to_dst,
    const char* utf8_path,
    char* err,
    int err_len);

/*
 * Warp each image by H_to_ref (n row-major 3x3 matrices, working-scale pixels
 * into the reference camera) and feather-blend. rotate_ccw90 places camera 1
 * on the left when the reference +Y axis points toward camera 6.
 */
SC_API int sc_stitch(
    const int* handles,
    const double* H_to_ref,
    int n,
    const char* utf8_out_path,
    int rotate_ccw90,
    ScCanvas* canvas,
    char* err,
    int err_len);

}
