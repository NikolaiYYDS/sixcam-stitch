#include "stitch_core.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace {

std::mutex g_mu;

struct Image {
    cv::Mat bgr;
    cv::Mat gray;
    int full_w = 0;
    int full_h = 0;
    std::vector<cv::Point2d> grid;
    std::vector<cv::KeyPoint> keys;
    cv::Mat desc;
    bool sift_ready = false;
};

std::vector<Image> g_images;

void set_err(char* err, int err_len, const std::string& msg) {
    if (err == nullptr || err_len <= 0) {
        return;
    }
    std::strncpy(err, msg.c_str(), static_cast<size_t>(err_len - 1));
    err[err_len - 1] = '\0';
}

std::wstring utf8_to_wide(const char* utf8) {
#ifdef _WIN32
    if (utf8 == nullptr || utf8[0] == '\0') {
        return {};
    }
    int n = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, nullptr, 0);
    if (n <= 0) {
        return {};
    }
    std::wstring out(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8, -1, out.data(), n);
    if (!out.empty() && out.back() == L'\0') {
        out.pop_back();
    }
    return out;
#else
    return std::wstring(utf8, utf8 + std::strlen(utf8));
#endif
}

std::vector<uchar> read_bytes(const char* utf8, std::string& err) {
#ifdef _WIN32
    std::wstring path = utf8_to_wide(utf8);
    FILE* file = _wfopen(path.c_str(), L"rb");
#else
    FILE* file = std::fopen(utf8, "rb");
#endif
    if (file == nullptr) {
        err = std::string("cannot open ") + utf8;
        return {};
    }
    if (std::fseek(file, 0, SEEK_END) != 0) {
        std::fclose(file);
        err = "seek failed";
        return {};
    }
    long bytes = std::ftell(file);
    if (bytes <= 0) {
        std::fclose(file);
        err = "empty image";
        return {};
    }
    std::rewind(file);
    std::vector<uchar> buf(static_cast<size_t>(bytes));
    size_t got = std::fread(buf.data(), 1, buf.size(), file);
    std::fclose(file);
    if (got != buf.size()) {
        err = "short read";
        return {};
    }
    return buf;
}

bool write_bytes(const char* utf8, const std::vector<uchar>& buf, std::string& err) {
#ifdef _WIN32
    std::wstring path = utf8_to_wide(utf8);
    FILE* file = _wfopen(path.c_str(), L"wb");
#else
    FILE* file = std::fopen(utf8, "wb");
#endif
    if (file == nullptr) {
        err = std::string("cannot write ") + utf8;
        return false;
    }
    size_t put = std::fwrite(buf.data(), 1, buf.size(), file);
    std::fclose(file);
    if (put != buf.size()) {
        err = "short write";
        return false;
    }
    return true;
}

bool save_bgr(const char* utf8, const cv::Mat& bgr, std::string& err) {
    std::vector<uchar> encoded;
    std::string ext = ".jpg";
    std::string path = utf8 == nullptr ? "" : utf8;
    auto dot = path.find_last_of('.');
    if (dot != std::string::npos) {
        ext = path.substr(dot);
    }
    if (!cv::imencode(ext, bgr, encoded)) {
        err = "imencode failed";
        return false;
    }
    return write_bytes(utf8, encoded, err);
}

Image* image_at(int handle) {
    if (handle < 0 || static_cast<size_t>(handle) >= g_images.size()) {
        return nullptr;
    }
    if (g_images[static_cast<size_t>(handle)].bgr.empty()) {
        return nullptr;
    }
    return &g_images[static_cast<size_t>(handle)];
}

void ensure_sift(Image& image) {
    if (image.sift_ready) {
        return;
    }
    // The circle grid is repetitive and moves between placements. Keep SIFT on
    // the static drawing so the homography is the camera-to-camera plane map.
    cv::Mat mask(image.gray.size(), CV_8U, cv::Scalar(255));
    if (image.grid.size() >= 4) {
        std::vector<cv::Point> pts;
        pts.reserve(image.grid.size());
        for (const auto& p : image.grid) {
            pts.emplace_back(static_cast<int>(std::lround(p.x)), static_cast<int>(std::lround(p.y)));
        }
        std::vector<cv::Point> hull;
        cv::convexHull(pts, hull);
        cv::fillConvexPoly(mask, hull, 0);
        const int rad = std::max(21, image.gray.cols / 35);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(rad, rad));
        cv::erode(mask, mask, kernel);
    }
    auto sift = cv::SIFT::create(4000);
    sift->detectAndCompute(image.gray, mask, image.keys, image.desc);
    image.sift_ready = true;
}

std::vector<cv::Point2d> detect_grid(const cv::Mat& gray) {
    const double rel = gray.cols / 5472.0;
    const double area_k = std::max(0.25, (rel / 0.2) * (rel / 0.2));
    cv::SimpleBlobDetector::Params params;
    params.filterByArea = true;
    params.minArea = static_cast<float>(20.0 * area_k);
    params.maxArea = static_cast<float>(8000.0 * area_k);
    params.filterByCircularity = true;
    params.minCircularity = 0.35f;
    params.filterByConvexity = false;
    params.filterByInertia = true;
    params.minInertiaRatio = 0.2f;
    params.filterByColor = true;
    params.blobColor = 0;
    auto detector = cv::SimpleBlobDetector::create(params);

    const int pad = std::max(40, static_cast<int>(std::lround(40.0 * rel / 0.2)));
    cv::Mat padded;
    cv::copyMakeBorder(gray, padded, pad, pad, pad, pad, cv::BORDER_REPLICATE);

    const int flags = cv::CALIB_CB_SYMMETRIC_GRID | cv::CALIB_CB_CLUSTERING;
    std::vector<cv::Point2d> best;
    int best_n = 0;
    for (int rows = 7; rows >= 3; --rows) {
        for (int cols = 7; cols >= 3; --cols) {
            const int n = rows * cols;
            if (n <= best_n || n < 12) {
                continue;
            }
            std::vector<cv::Point2f> centers;
            bool ok = cv::findCirclesGrid(padded, cv::Size(cols, rows), centers, flags, detector);
            if (!ok) {
                ok = cv::findCirclesGrid(padded, cv::Size(cols, rows), centers, flags);
            }
            if (!ok) {
                continue;
            }
            best.clear();
            best.reserve(centers.size());
            for (const auto& p : centers) {
                best.emplace_back(p.x - pad, p.y - pad);
            }
            best_n = n;
            if (rows == 7 && cols == 7) {
                return best;
            }
        }
    }
    return best;
}

void mat_to_h9(const cv::Mat& H, double* H9) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            H9[r * 3 + c] = H.at<double>(r, c);
        }
    }
}

cv::Mat h9_to_mat(const double* H9) {
    cv::Mat H(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            H.at<double>(r, c) = H9[r * 3 + c];
        }
    }
    return H;
}

}  // namespace

extern "C" {

int sc_open(const char* utf8_path, double scale, char* err, int err_len) {
    std::string message;
    auto bytes = read_bytes(utf8_path, message);
    if (bytes.empty()) {
        set_err(err, err_len, message);
        return -1;
    }
    cv::Mat color = cv::imdecode(bytes, cv::IMREAD_COLOR);
    if (color.empty()) {
        set_err(err, err_len, "imdecode failed");
        return -1;
    }
    Image image;
    image.full_w = color.cols;
    image.full_h = color.rows;
    if (std::abs(scale - 1.0) > 1e-9) {
        cv::resize(color, image.bgr, cv::Size(), scale, scale, cv::INTER_AREA);
    } else {
        image.bgr = color;
    }
    cv::cvtColor(image.bgr, image.gray, cv::COLOR_BGR2GRAY);
    image.grid = detect_grid(image.gray);

    std::lock_guard<std::mutex> lock(g_mu);
    g_images.push_back(std::move(image));
    return static_cast<int>(g_images.size() - 1);
}

void sc_close(int handle) {
    std::lock_guard<std::mutex> lock(g_mu);
    if (handle < 0 || static_cast<size_t>(handle) >= g_images.size()) {
        return;
    }
    g_images[static_cast<size_t>(handle)] = Image{};
}

int sc_size(int handle, int* width, int* height) {
    std::lock_guard<std::mutex> lock(g_mu);
    Image* image = image_at(handle);
    if (image == nullptr || width == nullptr || height == nullptr) {
        return -1;
    }
    *width = image->bgr.cols;
    *height = image->bgr.rows;
    return 0;
}

int sc_full_size(int handle, int* width, int* height) {
    std::lock_guard<std::mutex> lock(g_mu);
    Image* image = image_at(handle);
    if (image == nullptr || width == nullptr || height == nullptr) {
        return -1;
    }
    *width = image->full_w;
    *height = image->full_h;
    return 0;
}

int sc_thumb(int handle, int tw, int th, uint8_t* out) {
    std::lock_guard<std::mutex> lock(g_mu);
    Image* image = image_at(handle);
    if (image == nullptr || out == nullptr || tw <= 0 || th <= 0) {
        return -1;
    }
    cv::Mat thumb;
    cv::resize(image->gray, thumb, cv::Size(tw, th), 0, 0, cv::INTER_AREA);
    if (!thumb.isContinuous()) {
        thumb = thumb.clone();
    }
    std::memcpy(out, thumb.data, static_cast<size_t>(tw * th));
    return 0;
}

int sc_grid(int handle, double* xy, int max_points) {
    std::lock_guard<std::mutex> lock(g_mu);
    Image* image = image_at(handle);
    if (image == nullptr || xy == nullptr || max_points <= 0) {
        return -1;
    }
    const int n = std::min(max_points, static_cast<int>(image->grid.size()));
    for (int i = 0; i < n; ++i) {
        xy[i * 2] = image->grid[static_cast<size_t>(i)].x;
        xy[i * 2 + 1] = image->grid[static_cast<size_t>(i)].y;
    }
    return n;
}

int sc_sift_homography(
    const int* handles_a,
    int count_a,
    const int* handles_b,
    int count_b,
    double* H9,
    int* matches,
    int* inliers,
    char* err,
    int err_len) {
    if (handles_a == nullptr || handles_b == nullptr || H9 == nullptr || count_a <= 0 || count_b <= 0) {
        set_err(err, err_len, "bad sift arguments");
        return -1;
    }
    std::lock_guard<std::mutex> lock(g_mu);
    std::vector<cv::Point2f> pts_a;
    std::vector<cv::Point2f> pts_b;
    cv::BFMatcher matcher(cv::NORM_L2);

    for (int i = 0; i < count_a; ++i) {
        Image* a = image_at(handles_a[i]);
        if (a == nullptr) {
            set_err(err, err_len, "invalid handle in set A");
            return -1;
        }
        ensure_sift(*a);
        if (a->desc.empty()) {
            continue;
        }
        for (int j = 0; j < count_b; ++j) {
            Image* b = image_at(handles_b[j]);
            if (b == nullptr) {
                set_err(err, err_len, "invalid handle in set B");
                return -1;
            }
            ensure_sift(*b);
            if (b->desc.rows < 2 || a->desc.rows < 2) {
                continue;
            }
            std::vector<std::vector<cv::DMatch>> knn;
            matcher.knnMatch(a->desc, b->desc, knn, 2);
            for (const auto& pair : knn) {
                if (pair.size() < 2) {
                    continue;
                }
                if (pair[0].distance < 0.75f * pair[1].distance) {
                    pts_a.push_back(a->keys[static_cast<size_t>(pair[0].queryIdx)].pt);
                    pts_b.push_back(b->keys[static_cast<size_t>(pair[0].trainIdx)].pt);
                }
            }
        }
    }

    if (matches != nullptr) {
        *matches = static_cast<int>(pts_a.size());
    }
    if (pts_a.size() < 12) {
        set_err(err, err_len, "not enough SIFT matches");
        return -1;
    }
    cv::Mat inlier_mask;
    cv::Mat H = cv::findHomography(pts_a, pts_b, cv::RANSAC, 3.0, inlier_mask);
    if (H.empty()) {
        set_err(err, err_len, "findHomography failed");
        return -1;
    }
    int inl = 0;
    if (!inlier_mask.empty()) {
        inl = cv::countNonZero(inlier_mask);
    }
    if (inliers != nullptr) {
        *inliers = inl;
    }
    mat_to_h9(H, H9);
    return 0;
}

int sc_annulus_ncc(
    int handle_src,
    int handle_dst,
    const double* H_src_to_dst,
    const double* dst_xy,
    int n,
    double* score) {
    if (score == nullptr) {
        return -1;
    }
    *score = -1.0;
    std::lock_guard<std::mutex> lock(g_mu);
    Image* src = image_at(handle_src);
    Image* dst = image_at(handle_dst);
    if (src == nullptr || dst == nullptr || H_src_to_dst == nullptr || dst_xy == nullptr || n < 4) {
        return -1;
    }
    cv::Mat H = h9_to_mat(H_src_to_dst);
    cv::Mat warped;
    cv::warpPerspective(src->gray, warped, H, dst->gray.size());
    std::vector<cv::Point> hull_pts;
    hull_pts.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        hull_pts.emplace_back(static_cast<int>(std::lround(dst_xy[i * 2])), static_cast<int>(std::lround(dst_xy[i * 2 + 1])));
    }
    std::vector<cv::Point> hull;
    cv::convexHull(hull_pts, hull);
    cv::Mat board(dst->gray.size(), CV_8U, cv::Scalar(0));
    cv::fillConvexPoly(board, hull, 255);
    const int gap = std::max(9, std::min(dst->gray.cols, dst->gray.rows) / 80);
    const int ring = std::max(gap * 4, 36);
    cv::Mat inner, outer;
    cv::dilate(board, inner, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(gap, gap)));
    cv::dilate(board, outer, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ring, ring)));
    cv::Mat annulus;
    cv::subtract(outer, inner, annulus);
    double sum_a = 0, sum_b = 0;
    int count = 0;
    for (int y = 0; y < annulus.rows; y += 2) {
        const uchar* am = annulus.ptr<uchar>(y);
        const uchar* wa = warped.ptr<uchar>(y);
        const uchar* db = dst->gray.ptr<uchar>(y);
        for (int x = 0; x < annulus.cols; x += 2) {
            if (am[x] == 0 || wa[x] == 0) {
                continue;
            }
            sum_a += wa[x];
            sum_b += db[x];
            ++count;
        }
    }
    if (count < 200) {
        return 0;
    }
    const double mean_a = sum_a / count;
    const double mean_b = sum_b / count;
    double num = 0, na = 0, nb = 0;
    for (int y = 0; y < annulus.rows; y += 2) {
        const uchar* am = annulus.ptr<uchar>(y);
        const uchar* wa = warped.ptr<uchar>(y);
        const uchar* db = dst->gray.ptr<uchar>(y);
        for (int x = 0; x < annulus.cols; x += 2) {
            if (am[x] == 0 || wa[x] == 0) {
                continue;
            }
            const double da = wa[x] - mean_a;
            const double dbv = db[x] - mean_b;
            num += da * dbv;
            na += da * da;
            nb += dbv * dbv;
        }
    }
    const double denom = std::sqrt(na) * std::sqrt(nb);
    if (denom < 1e-6) {
        return 0;
    }
    *score = num / denom;
    return 0;
}

int sc_fit_homography(
    const double* src_xy,
    const double* dst_xy,
    int n,
    double ransac_px,
    double* H9,
    int* inliers) {
    if (src_xy == nullptr || dst_xy == nullptr || H9 == nullptr || n < 4) {
        return -1;
    }
    std::vector<cv::Point2f> src;
    std::vector<cv::Point2f> dst;
    src.reserve(static_cast<size_t>(n));
    dst.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        src.emplace_back(static_cast<float>(src_xy[i * 2]), static_cast<float>(src_xy[i * 2 + 1]));
        dst.emplace_back(static_cast<float>(dst_xy[i * 2]), static_cast<float>(dst_xy[i * 2 + 1]));
    }
    cv::Mat inlier_mask;
    cv::Mat H;
    if (ransac_px > 0.0) {
        H = cv::findHomography(src, dst, cv::RANSAC, ransac_px, inlier_mask);
    } else {
        H = cv::findHomography(src, dst, 0);
    }
    if (H.empty()) {
        return -1;
    }
    if (inliers != nullptr) {
        *inliers = inlier_mask.empty() ? n : cv::countNonZero(inlier_mask);
    }
    mat_to_h9(H, H9);
    return 0;
}

int sc_save_overlap(
    int handle_src,
    int handle_dst,
    const double* H_src_to_dst,
    const char* utf8_path,
    char* err,
    int err_len) {
    std::lock_guard<std::mutex> lock(g_mu);
    Image* src = image_at(handle_src);
    Image* dst = image_at(handle_dst);
    if (src == nullptr || dst == nullptr || H_src_to_dst == nullptr) {
        set_err(err, err_len, "bad overlap arguments");
        return -1;
    }
    cv::Mat H = h9_to_mat(H_src_to_dst);
    cv::Mat warped;
    cv::warpPerspective(src->bgr, warped, H, dst->bgr.size());
    cv::Mat blend;
    cv::addWeighted(warped, 0.5, dst->bgr, 0.5, 0, blend);
    std::string message;
    if (!save_bgr(utf8_path, blend, message)) {
        set_err(err, err_len, message);
        return -1;
    }
    return 0;
}

int sc_stitch(
    const int* handles,
    const double* H_to_ref,
    int n,
    const char* utf8_out_path,
    int rotate_ccw90,
    ScCanvas* canvas,
    char* err,
    int err_len) {
    if (handles == nullptr || H_to_ref == nullptr || n <= 0) {
        set_err(err, err_len, "bad stitch arguments");
        return -1;
    }
    std::lock_guard<std::mutex> lock(g_mu);
    std::vector<cv::Point2f> corners;
    std::vector<cv::Mat> homographies;
    homographies.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        Image* image = image_at(handles[i]);
        if (image == nullptr) {
            set_err(err, err_len, "invalid stitch handle");
            return -1;
        }
        cv::Mat H = h9_to_mat(H_to_ref + i * 9);
        homographies.push_back(H);
        std::vector<cv::Point2f> box = {
            {0.f, 0.f},
            {static_cast<float>(image->bgr.cols), 0.f},
            {static_cast<float>(image->bgr.cols), static_cast<float>(image->bgr.rows)},
            {0.f, static_cast<float>(image->bgr.rows)},
        };
        std::vector<cv::Point2f> warped;
        cv::perspectiveTransform(box, warped, H);
        corners.insert(corners.end(), warped.begin(), warped.end());
    }

    float min_x = corners[0].x;
    float min_y = corners[0].y;
    float max_x = corners[0].x;
    float max_y = corners[0].y;
    for (const auto& p : corners) {
        min_x = std::min(min_x, p.x);
        min_y = std::min(min_y, p.y);
        max_x = std::max(max_x, p.x);
        max_y = std::max(max_y, p.y);
    }
    const float span_x = max_x - min_x;
    const float span_y = max_y - min_y;
    if (std::max(span_x, span_y) > 20000.f) {
        set_err(err, err_len, "canvas exploded; homography is unstable");
        return -1;
    }
    const double fit = std::min(1.0, 6000.0 / static_cast<double>(std::max(span_x, span_y)));
    const int cw = std::max(1, static_cast<int>(std::ceil(span_x * fit)));
    const int ch = std::max(1, static_cast<int>(std::ceil(span_y * fit)));
    cv::Mat T = (cv::Mat_<double>(3, 3) << fit, 0, -static_cast<double>(min_x) * fit,
                 0, fit, -static_cast<double>(min_y) * fit,
                 0, 0, 1);

    cv::Mat acc(ch, cw, CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat weight(ch, cw, CV_32F, cv::Scalar(0));
    for (int i = 0; i < n; ++i) {
        Image* image = image_at(handles[i]);
        cv::Mat H = T * homographies[static_cast<size_t>(i)];
        cv::Mat warped;
        cv::warpPerspective(image->bgr, warped, H, cv::Size(cw, ch));
        cv::Mat mask;
        cv::warpPerspective(cv::Mat(image->bgr.size(), CV_8U, cv::Scalar(255)), mask, H, cv::Size(cw, ch));
        cv::threshold(mask, mask, 127, 255, cv::THRESH_BINARY);
        cv::Mat dist;
        cv::distanceTransform(mask, dist, cv::DIST_L2, 3);
        double max_dist = 0;
        cv::minMaxLoc(dist, nullptr, &max_dist);
        if (max_dist > 0) {
            dist /= max_dist;
        }
        cv::Mat warped_f;
        warped.convertTo(warped_f, CV_32FC3);
        std::vector<cv::Mat> channels(3);
        cv::split(warped_f, channels);
        for (int c = 0; c < 3; ++c) {
            channels[static_cast<size_t>(c)] = channels[static_cast<size_t>(c)].mul(dist);
        }
        cv::Mat weighted;
        cv::merge(channels, weighted);
        acc += weighted;
        weight += dist;
    }

    cv::Mat weight3;
    cv::Mat weight_safe;
    cv::max(weight, 1e-6, weight_safe);
    cv::merge(std::vector<cv::Mat>{weight_safe, weight_safe, weight_safe}, weight3);
    cv::Mat mosaic_f;
    cv::divide(acc, weight3, mosaic_f);
    cv::Mat mosaic;
    mosaic_f.convertTo(mosaic, CV_8UC3);
    if (rotate_ccw90) {
        cv::rotate(mosaic, mosaic, cv::ROTATE_90_COUNTERCLOCKWISE);
    }
    std::string message;
    if (!save_bgr(utf8_out_path, mosaic, message)) {
        set_err(err, err_len, message);
        return -1;
    }
    if (canvas != nullptr) {
        canvas->min_x = min_x;
        canvas->min_y = min_y;
        canvas->fit = fit;
        canvas->width_before_rot = cw;
        canvas->height_before_rot = ch;
        canvas->out_width = mosaic.cols;
        canvas->out_height = mosaic.rows;
    }
    return 0;
}

}  // extern "C"
