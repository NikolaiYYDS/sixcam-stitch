//! Six-camera plane stitcher.
//!
//! Rust owns the capture layout, camera identity, homography chain, and the
//! calibration file. OpenCV (C++) detects the 7x7 circle grid, matches the
//! drawing, and warps the panorama.

use std::collections::HashMap;
use std::env;
use std::ffi::{CStr, CString};
use std::fs;
use std::os::raw::c_char;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const TILDE: char = '\u{ff5e}';
const BORDER_NCC: f64 = 0.74;

#[repr(C)]
struct ScCanvas {
    min_x: f64,
    min_y: f64,
    fit: f64,
    width_before_rot: i32,
    height_before_rot: i32,
    out_width: i32,
    out_height: i32,
}

extern "C" {
    fn sc_open(path: *const c_char, scale: f64, err: *mut c_char, err_len: i32) -> i32;
    fn sc_close(handle: i32);
    fn sc_full_size(handle: i32, width: *mut i32, height: *mut i32) -> i32;
    fn sc_thumb(handle: i32, tw: i32, th: i32, out: *mut u8) -> i32;
    fn sc_grid(handle: i32, xy: *mut f64, max_points: i32) -> i32;
    fn sc_sift_homography(
        handles_a: *const i32,
        count_a: i32,
        handles_b: *const i32,
        count_b: i32,
        h9: *mut f64,
        matches: *mut i32,
        inliers: *mut i32,
        err: *mut c_char,
        err_len: i32,
    ) -> i32;
    fn sc_annulus_ncc(
        handle_src: i32,
        handle_dst: i32,
        h_src_to_dst: *const f64,
        dst_xy: *const f64,
        n: i32,
        score: *mut f64,
    ) -> i32;
    fn sc_fit_homography(
        src_xy: *const f64,
        dst_xy: *const f64,
        n: i32,
        ransac_px: f64,
        h9: *mut f64,
        inliers: *mut i32,
    ) -> i32;
    fn sc_save_overlap(
        handle_src: i32,
        handle_dst: i32,
        h_src_to_dst: *const f64,
        path: *const c_char,
        err: *mut c_char,
        err_len: i32,
    ) -> i32;
    fn sc_stitch(
        handles: *const i32,
        h_to_ref: *const f64,
        n: i32,
        path: *const c_char,
        rotate_ccw90: i32,
        canvas: *mut ScCanvas,
        err: *mut c_char,
        err_len: i32,
    ) -> i32;
}

#[cfg(windows)]
extern "system" {
    fn SetConsoleOutputCP(code: u32) -> i32;
}

type Mat3 = [f64; 9];

struct Shot {
    path: PathBuf,
    folder: String,
    stem: String,
    handle: i32,
    full_w: i32,
    full_h: i32,
    grid: Vec<[f64; 2]>,
    camera: u32,
}

struct PairCalib {
    src: u32,
    dst: u32,
    method: &'static str,
    sift_matches: i32,
    sift_inliers: i32,
    circle_points: usize,
    circle_rms_px: Option<f64>,
    board_sift_median_px: Option<f64>,
    h_work: Mat3,
}

fn main() -> ExitCode {
    #[cfg(windows)]
    unsafe {
        SetConsoleOutputCP(65001);
    }
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("错误: {err}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<(), String> {
    let mut data: Option<String> = None;
    let mut out: Option<String> = None;
    let mut scale = 0.30_f64;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--data" => data = Some(args.next().ok_or("--data 需要路径")?),
            "--out" => out = Some(args.next().ok_or("--out 需要路径")?),
            "--scale" => {
                scale = args
                    .next()
                    .ok_or("--scale 需要数值")?
                    .parse()
                    .map_err(|_| "--scale 不是数字".to_string())?;
            }
            "--help" | "-h" => {
                println!(
                    "用法: sixcam-stitch [--data 标定数据目录] [--out 输出目录] [--scale 0.3]\n\
                     相机 1 在左，相机 6 在右。相机 2 与其余相机约成 90 度，仍按桌面平面拼接。"
                );
                return Ok(());
            }
            other => return Err(format!("未知参数 {other}")),
        }
    }
    if !(0.05..=1.0).contains(&scale) {
        return Err("scale 应在 0.05 到 1 之间".into());
    }

    let data_dir = resolve_data(data)?;
    let out_dir = PathBuf::from(out.unwrap_or_else(|| "result".into()));
    fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
    println!("数据目录: {}", data_dir.display());
    println!("输出目录: {}", out_dir.display());
    println!("工作尺度: {scale}");

    let mut shots = load_shots(&data_dir, scale)?;
    assign_cameras(&mut shots)?;
    for shot in &shots {
        println!(
            "  相机{}  {}  圆点 {}",
            shot.camera,
            shot.path.file_name().unwrap_or_default().to_string_lossy(),
            shot.grid.len()
        );
    }

    let by_cam = group_by_camera(&shots);
    let primary = primary_index(&shots, &by_cam)?;
    for cam in 1..=6 {
        let shot = &shots[primary[&cam]];
        println!("主图 相机{cam}: {}", shot.stem);
    }

    let mut pairs = Vec::new();
    for src in 2u32..=6 {
        let dst = src - 1;
        let pair = calibrate_pair(&shots, &by_cam, src, dst, scale)?;
        println!(
            "相机{src}->相机{dst}: {}  SIFT内点 {}  圆点 {}{}",
            pair.method,
            pair.sift_inliers,
            pair.circle_points,
            pair.circle_rms_px
                .map(|e| format!("  圆点RMS {e:.3}px"))
                .unwrap_or_default()
        );
        let overlap = shared_folder(&shots, &by_cam, src, dst);
        if let Some((ia, ib)) = overlap {
            let path = out_dir.join(format!("overlap_{src}_on_{dst}.jpg"));
            save_overlap(shots[ia].handle, shots[ib].handle, &pair.h_work, &path)?;
        }
        pairs.push(pair);
    }

    let mut h_to_1: HashMap<u32, Mat3> = HashMap::new();
    h_to_1.insert(1, identity());
    for src in 2u32..=6 {
        let pair = pairs.iter().find(|p| p.src == src).unwrap();
        let prev = h_to_1[&(src - 1)];
        h_to_1.insert(src, mul(&prev, &pair.h_work));
    }

    let handles: Vec<i32> = (1u32..=6).map(|cam| shots[primary[&cam]].handle).collect();
    let mut packed = Vec::with_capacity(6 * 9);
    for cam in 1u32..=6 {
        packed.extend_from_slice(&h_to_1[&cam]);
    }
    let mosaic_path = out_dir.join("mosaic.jpg");
    let canvas = stitch(&handles, &packed, &mosaic_path)?;
    println!(
        "拼接图 {}x{} -> {}",
        canvas.out_width,
        canvas.out_height,
        mosaic_path.display()
    );

    let sensor_w = shots[0].full_w;
    let sensor_h = shots[0].full_h;
    let calib_path = out_dir.join("calibration.json");
    write_calibration(
        &calib_path,
        &shots,
        &pairs,
        &h_to_1,
        &primary,
        scale,
        sensor_w,
        sensor_h,
        &canvas,
    )?;
    println!("标定文件 {}", calib_path.display());

    for shot in &shots {
        unsafe { sc_close(shot.handle) };
    }
    Ok(())
}

fn resolve_data(arg: Option<String>) -> Result<PathBuf, String> {
    if let Some(path) = arg {
        let path = PathBuf::from(path);
        if path.is_dir() {
            return Ok(path);
        }
        return Err(format!("数据目录不存在: {}", path.display()));
    }
    let cwd = env::current_dir().map_err(|e| e.to_string())?;
    let candidates = [
        cwd.join("260924标定数据"),
        cwd.join("..").join("260924标定数据"),
    ];
    for path in &candidates {
        if path.is_dir() {
            return Ok(path.clone());
        }
    }
    Err("找不到 260924标定数据，请用 --data 指定".into())
}

fn load_shots(data_dir: &Path, scale: f64) -> Result<Vec<Shot>, String> {
    let mut paths = Vec::new();
    let entries = fs::read_dir(data_dir).map_err(|e| format!("读取数据目录失败: {e}"))?;
    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let folder_path = entry.path();
        if !folder_path.is_dir() {
            continue;
        }
        let folder = entry.file_name().to_string_lossy().into_owned();
        let files = fs::read_dir(&folder_path).map_err(|e| e.to_string())?;
        for file in files {
            let file = file.map_err(|e| e.to_string())?;
            let path = file.path();
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if ext == "tif" || ext == "tiff" {
                paths.push((folder.clone(), path));
            }
        }
    }
    paths.sort_by(|a, b| a.1.cmp(&b.1));
    if paths.is_empty() {
        return Err("目录里没有 tif 图像".into());
    }
    println!("读取 {} 张图", paths.len());
    let mut shots = Vec::new();
    for (folder, path) in paths {
        let handle = open_image(&path, scale)?;
        let (full_w, full_h) = full_size(handle)?;
        let grid = read_grid(handle)?;
        let stem = path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        shots.push(Shot {
            path,
            folder,
            stem,
            handle,
            full_w,
            full_h,
            grid,
            camera: 0,
        });
    }
    Ok(shots)
}

fn assign_cameras(shots: &mut [Shot]) -> Result<(), String> {
    let thumbs: Vec<Vec<u8>> = shots
        .iter()
        .map(|s| thumbnail(s.handle, 160, 107))
        .collect::<Result<_, _>>()?;
    let n = shots.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if border_ncc(&thumbs[i], &thumbs[j], 160, 107) >= BORDER_NCC {
                let ri = find(&mut parent, i);
                let rj = find(&mut parent, j);
                parent[ri] = rj;
            }
        }
    }
    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        clusters.entry(find(&mut parent, i)).or_default().push(i);
    }
    println!("识别到 {} 个相机视角", clusters.len());

    let mut by_folder: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, shot) in shots.iter().enumerate() {
        by_folder.entry(shot.folder.clone()).or_default().push(i);
    }
    let mut cluster_cam: HashMap<usize, u32> = HashMap::new();
    for (folder, idxs) in &by_folder {
        if idxs.len() == 1 {
            if let Some(num) = station_number(folder) {
                let root = find(&mut parent, idxs[0]);
                cluster_cam.insert(root, num);
            }
        }
    }
    for (root, members) in &clusters {
        if cluster_cam.contains_key(root) {
            continue;
        }
        let in_cam1 = members.iter().any(|i| station_number(&shots[*i].folder) == Some(1));
        if in_cam1 {
            cluster_cam.insert(*root, 1);
        }
    }
    if cluster_cam.len() != 6 || (1..=6).any(|c| !cluster_cam.values().any(|v| *v == c)) {
        return Err(format!("相机编号不完整: {cluster_cam:?}"));
    }
    for (root, members) in &clusters {
        let cam = cluster_cam[root];
        for i in members {
            shots[*i].camera = cam;
        }
    }
    Ok(())
}

fn station_number(name: &str) -> Option<u32> {
    if name.contains(TILDE) {
        return None;
    }
    let digits: String = name.chars().filter(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

fn group_by_camera(shots: &[Shot]) -> HashMap<u32, Vec<usize>> {
    let mut map = HashMap::new();
    for (i, shot) in shots.iter().enumerate() {
        map.entry(shot.camera).or_insert_with(Vec::new).push(i);
    }
    map
}

fn primary_index(shots: &[Shot], by_cam: &HashMap<u32, Vec<usize>>) -> Result<HashMap<u32, usize>, String> {
    let mut primary = HashMap::new();
    for cam in 1u32..=6 {
        let members = by_cam.get(&cam).ok_or(format!("缺少相机 {cam}"))?;
        let station = members
            .iter()
            .copied()
            .find(|i| station_number(&shots[*i].folder) == Some(cam));
        let idx = station.unwrap_or(members[0]);
        primary.insert(cam, idx);
    }
    Ok(primary)
}

fn calibrate_pair(
    shots: &[Shot],
    by_cam: &HashMap<u32, Vec<usize>>,
    src: u32,
    dst: u32,
    scale: f64,
) -> Result<PairCalib, String> {
    let a: Vec<i32> = by_cam[&src].iter().map(|i| shots[*i].handle).collect();
    let b: Vec<i32> = by_cam[&dst].iter().map(|i| shots[*i].handle).collect();
    let (h_sift, matches, inliers) = sift_homography(&a, &b)?;

    let mut circle_src = Vec::new();
    let mut circle_dst = Vec::new();
    let folders_src: Vec<&str> = by_cam[&src].iter().map(|i| shots[*i].folder.as_str()).collect();
    for i in &by_cam[&dst] {
        let folder = shots[*i].folder.as_str();
        if !folders_src.iter().any(|f| *f == folder) {
            continue;
        }
        let ia = by_cam[&src]
            .iter()
            .copied()
            .find(|k| shots[*k].folder == folder)
            .unwrap();
        let (ss, dd) = match_circles(&h_sift, &shots[ia].grid, &shots[*i].grid, 60.0 * scale);
        circle_src.extend(ss);
        circle_dst.extend(dd);
    }

    let mut method = "sift_background";
    let mut h_use = h_sift;
    let mut rms = None;
    let mut agree = None;
    let mut circle_points = circle_src.len();
    if circle_src.len() >= 12 && grid_scale_ok(&h_sift, &circle_src) {
        let (h_board, _inl) = fit_homography(&circle_src, &circle_dst, 1.5)?;
        let err = mean_transfer_error(&h_board, &circle_src, &circle_dst);
        let median = median_disagreement(&h_board, &h_sift, &circle_src);
        println!("  圆点与背景单应中位差 {median:.2}px（工作尺度）");
        rms = Some(err);
        agree = Some(median);
        if median < 40.0 * scale && err < 15.0 * scale && grid_scale_ok(&h_board, &circle_src) {
            h_use = h_board;
            method = "circle_grid";
        }
    }

    let sift_ok = shared_grids_ok(shots, by_cam, src, dst, &h_use);
    if method != "circle_grid" || !sift_ok {
        if let Some(found) = search_grid_homography(shots, by_cam, src, dst)? {
            println!(
                "  圆点网格搜索 NCC {:.3}  点数 {}  RMS {:.3}px",
                found.ncc, found.points, found.rms
            );
            if found.ncc > 0.2 {
                h_use = found.h;
                method = "circle_grid_search";
                rms = Some(found.rms);
                circle_points = found.points;
                agree = None;
            }
        }
    }
    if !shared_grids_ok(shots, by_cam, src, dst, &h_use) && method == "sift_background" {
        return Err(format!("相机{src}->相机{dst} 的单应不稳定，圆点也无法对齐"));
    }

    Ok(PairCalib {
        src,
        dst,
        method,
        sift_matches: matches,
        sift_inliers: inliers,
        circle_points,
        circle_rms_px: rms,
        board_sift_median_px: agree,
        h_work: h_use,
    })
}

struct GridHit {
    h: Mat3,
    ncc: f64,
    points: usize,
    rms: f64,
}

fn shared_grids_ok(shots: &[Shot], by_cam: &HashMap<u32, Vec<usize>>, src: u32, dst: u32, h: &Mat3) -> bool {
    for i in &by_cam[&src] {
        let Some(j) = by_cam[&dst].iter().copied().find(|j| shots[*j].folder == shots[*i].folder) else {
            continue;
        };
        if shots[*i].grid.len() < 12 || shots[j].grid.len() < 12 {
            continue;
        }
        if !grid_scale_ok(h, &shots[*i].grid) {
            return false;
        }
    }
    true
}

fn grid_scale_ok(h: &Mat3, src: &[[f64; 2]]) -> bool {
    if src.len() < 4 {
        return true;
    }
    let proj: Vec<[f64; 2]> = src.iter().copied().map(|p| apply_h(h, p)).collect();
    if proj.iter().any(|p| !p[0].is_finite() || !p[1].is_finite()) {
        return false;
    }
    let ratio = bbox_area(&proj) / bbox_area(src).max(1.0);
    (0.25..4.0).contains(&ratio)
}

fn bbox_area(pts: &[[f64; 2]]) -> f64 {
    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;
    for p in pts {
        min_x = min_x.min(p[0]);
        min_y = min_y.min(p[1]);
        max_x = max_x.max(p[0]);
        max_y = max_y.max(p[1]);
    }
    (max_x - min_x).max(0.0) * (max_y - min_y).max(0.0)
}

fn search_grid_homography(
    shots: &[Shot],
    by_cam: &HashMap<u32, Vec<usize>>,
    src: u32,
    dst: u32,
) -> Result<Option<GridHit>, String> {
    let mut best: Option<GridHit> = None;
    for i in &by_cam[&src] {
        let Some(j) = by_cam[&dst].iter().copied().find(|j| shots[*j].folder == shots[*i].folder) else {
            continue;
        };
        let Some(gs) = reshape_grid(&shots[*i].grid) else { continue };
        let Some(gd) = reshape_grid(&shots[j].grid) else { continue };
        for vs in grid_variants(&gs) {
            for vd in grid_variants(&gd) {
                let (small, large, small_is_src) = if vs.len() * vs[0].len() <= vd.len() * vd[0].len() {
                    (&vs, &vd, true)
                } else {
                    (&vd, &vs, false)
                };
                if small.len() > large.len() || small[0].len() > large[0].len() {
                    continue;
                }
                let row_off = large.len() - small.len();
                let col_off = large[0].len() - small[0].len();
                for ro in 0..=row_off {
                    for co in 0..=col_off {
                        let mut src_pts = Vec::new();
                        let mut dst_pts = Vec::new();
                        for r in 0..small.len() {
                            for c in 0..small[0].len() {
                                let s = small[r][c];
                                let l = large[ro + r][co + c];
                                if small_is_src {
                                    src_pts.push(s);
                                    dst_pts.push(l);
                                } else {
                                    src_pts.push(l);
                                    dst_pts.push(s);
                                }
                            }
                        }
                        let Ok((h, _)) = fit_homography(&src_pts, &dst_pts, 0.0) else { continue };
                        if !grid_scale_ok(&h, &src_pts) {
                            continue;
                        }
                        let ncc = annulus_ncc(shots[*i].handle, shots[j].handle, &h, &dst_pts)?;
                        let better = best.as_ref().map(|b| ncc > b.ncc).unwrap_or(true);
                        if better {
                            let rms = mean_transfer_error(&h, &src_pts, &dst_pts);
                            best = Some(GridHit { h, ncc, points: src_pts.len(), rms });
                        }
                    }
                }
            }
        }
    }
    Ok(best)
}

fn reshape_grid(pts: &[[f64; 2]]) -> Option<Vec<Vec<[f64; 2]>>> {
    let n = pts.len();
    let mut best: Option<(f64, Vec<Vec<[f64; 2]>>)> = None;
    for rows in 3..=7 {
        if rows == 0 || n % rows != 0 {
            continue;
        }
        let cols = n / rows;
        if !(3..=7).contains(&cols) {
            continue;
        }
        let mut grid = vec![vec![[0.0, 0.0]; cols]; rows];
        for r in 0..rows {
            for c in 0..cols {
                grid[r][c] = pts[r * cols + c];
            }
        }
        let cv = spacing_cv(&grid);
        if best.as_ref().map(|(s, _)| cv < *s).unwrap_or(true) {
            best = Some((cv, grid));
        }
    }
    let (cv, grid) = best?;
    if cv < 0.2 { Some(grid) } else { None }
}

fn spacing_cv(grid: &[Vec<[f64; 2]>]) -> f64 {
    let mut dists = Vec::new();
    for row in grid {
        for w in row.windows(2) {
            dists.push(hypot(w[0], w[1]));
        }
    }
    let cols = grid[0].len();
    for c in 0..cols {
        for r in 0..grid.len() - 1 {
            dists.push(hypot(grid[r][c], grid[r + 1][c]));
        }
    }
    if dists.is_empty() {
        return f64::MAX;
    }
    let mean = dists.iter().sum::<f64>() / dists.len() as f64;
    if mean < 1e-6 {
        return f64::MAX;
    }
    let var = dists.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / dists.len() as f64;
    var.sqrt() / mean
}

fn grid_variants(grid: &[Vec<[f64; 2]>]) -> Vec<Vec<Vec<[f64; 2]>>> {
    let mut out = Vec::new();
    let mut g = grid.to_vec();
    for _ in 0..4 {
        out.push(g.clone());
        out.push(flip_cols(&g));
        g = rot90(&g);
    }
    out
}

fn rot90(grid: &[Vec<[f64; 2]>]) -> Vec<Vec<[f64; 2]>> {
    let rows = grid.len();
    let cols = grid[0].len();
    let mut out = vec![vec![[0.0, 0.0]; rows]; cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c][rows - 1 - r] = grid[r][c];
        }
    }
    out
}

fn flip_cols(grid: &[Vec<[f64; 2]>]) -> Vec<Vec<[f64; 2]>> {
    grid.iter().map(|row| {
        let mut row = row.clone();
        row.reverse();
        row
    }).collect()
}

fn annulus_ncc(src: i32, dst: i32, h: &Mat3, dst_pts: &[[f64; 2]]) -> Result<f64, String> {
    let flat = flatten(dst_pts);
    let mut score = -1.0;
    let rc = unsafe {
        sc_annulus_ncc(src, dst, h.as_ptr(), flat.as_ptr(), dst_pts.len() as i32, &mut score)
    };
    if rc != 0 {
        return Err("重叠评分失败".into());
    }
    Ok(score)
}

fn shared_folder(shots: &[Shot], by_cam: &HashMap<u32, Vec<usize>>, src: u32, dst: u32) -> Option<(usize, usize)> {
    for i in &by_cam[&src] {
        if let Some(j) = by_cam[&dst].iter().copied().find(|j| shots[*j].folder == shots[*i].folder) {
            return Some((*i, j));
        }
    }
    None
}

fn border_ncc(a: &[u8], b: &[u8], w: i32, h: i32) -> f64 {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let on_border = y < 12 || y >= h - 12 || x < 16 || x >= w - 16;
            if !on_border {
                continue;
            }
            let i = (y * w + x) as usize;
            xs.push(a[i] as f64);
            ys.push(b[i] as f64);
        }
    }
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..xs.len() {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        num += dx * dy;
        na += dx * dx;
        nb += dy * dy;
    }
    num / (na.sqrt() * nb.sqrt() + 1e-6)
}

fn match_circles(h: &Mat3, src: &[[f64; 2]], dst: &[[f64; 2]], max_dist: f64) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    if src.is_empty() || dst.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let proj: Vec<[f64; 2]> = src.iter().copied().map(|p| apply_h(h, p)).collect();
    let mut order: Vec<usize> = (0..src.len()).collect();
    order.sort_by(|&i, &j| {
        let di = dst.iter().map(|q| hypot(proj[i], *q)).fold(f64::MAX, f64::min);
        let dj = dst.iter().map(|q| hypot(proj[j], *q)).fold(f64::MAX, f64::min);
        di.partial_cmp(&dj).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut used = vec![false; dst.len()];
    let mut kept_src = Vec::new();
    let mut kept_dst = Vec::new();
    for i in order {
        let mut best_j = None;
        let mut best_d = f64::MAX;
        for (j, q) in dst.iter().enumerate() {
            if used[j] {
                continue;
            }
            let d = hypot(proj[i], *q);
            if d < best_d {
                best_d = d;
                best_j = Some(j);
            }
        }
        if let Some(j) = best_j {
            if best_d <= max_dist {
                used[j] = true;
                kept_src.push(src[i]);
                kept_dst.push(dst[j]);
            }
        }
    }
    (kept_src, kept_dst)
}

fn mean_transfer_error(h: &Mat3, src: &[[f64; 2]], dst: &[[f64; 2]]) -> f64 {
    let mut sum = 0.0;
    for (p, q) in src.iter().zip(dst.iter()) {
        sum += hypot(apply_h(h, *p), *q);
    }
    sum / src.len() as f64
}

fn median_disagreement(a: &Mat3, b: &Mat3, pts: &[[f64; 2]]) -> f64 {
    let mut d: Vec<f64> = pts
        .iter()
        .map(|p| hypot(apply_h(a, *p), apply_h(b, *p)))
        .collect();
    d.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    d[d.len() / 2]
}

fn identity() -> Mat3 {
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
}

fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut c = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            c[i * 3 + j] =
                a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
        }
    }
    c
}

fn apply_h(h: &Mat3, p: [f64; 2]) -> [f64; 2] {
    let x = h[0] * p[0] + h[1] * p[1] + h[2];
    let y = h[3] * p[0] + h[4] * p[1] + h[5];
    let w = h[6] * p[0] + h[7] * p[1] + h[8];
    [x / w, y / w]
}

fn hypot(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

fn to_full(h: &Mat3, scale: f64) -> Mat3 {
    [
        h[0],
        h[1],
        h[2] / scale,
        h[3],
        h[4],
        h[5] / scale,
        h[6] * scale,
        h[7] * scale,
        h[8],
    ]
}

fn open_image(path: &Path, scale: f64) -> Result<i32, String> {
    let cpath = cstring(path)?;
    let mut err = [0 as c_char; 512];
    let handle = unsafe { sc_open(cpath.as_ptr(), scale, err.as_mut_ptr(), err.len() as i32) };
    if handle < 0 {
        return Err(format!("打开 {} 失败: {}", path.display(), cerr(&err)));
    }
    Ok(handle)
}

fn full_size(handle: i32) -> Result<(i32, i32), String> {
    let mut w = 0;
    let mut h = 0;
    let rc = unsafe { sc_full_size(handle, &mut w, &mut h) };
    if rc != 0 {
        return Err("读取原始尺寸失败".into());
    }
    Ok((w, h))
}

fn thumbnail(handle: i32, tw: i32, th: i32) -> Result<Vec<u8>, String> {
    let mut buf = vec![0u8; (tw * th) as usize];
    let rc = unsafe { sc_thumb(handle, tw, th, buf.as_mut_ptr()) };
    if rc != 0 {
        return Err("缩略图失败".into());
    }
    Ok(buf)
}

fn read_grid(handle: i32) -> Result<Vec<[f64; 2]>, String> {
    let mut xy = vec![0f64; 49 * 2];
    let n = unsafe { sc_grid(handle, xy.as_mut_ptr(), 49) };
    if n < 0 {
        return Err("圆点检测失败".into());
    }
    let mut pts = Vec::with_capacity(n as usize);
    for i in 0..n as usize {
        pts.push([xy[i * 2], xy[i * 2 + 1]]);
    }
    Ok(pts)
}

fn sift_homography(a: &[i32], b: &[i32]) -> Result<(Mat3, i32, i32), String> {
    let mut h = [0f64; 9];
    let mut matches = 0;
    let mut inliers = 0;
    let mut err = [0 as c_char; 512];
    let rc = unsafe {
        sc_sift_homography(
            a.as_ptr(),
            a.len() as i32,
            b.as_ptr(),
            b.len() as i32,
            h.as_mut_ptr(),
            &mut matches,
            &mut inliers,
            err.as_mut_ptr(),
            err.len() as i32,
        )
    };
    if rc != 0 {
        return Err(format!("SIFT 单应失败: {}", cerr(&err)));
    }
    Ok((h, matches, inliers))
}

fn fit_homography(src: &[[f64; 2]], dst: &[[f64; 2]], ransac: f64) -> Result<(Mat3, i32), String> {
    let src_flat = flatten(src);
    let dst_flat = flatten(dst);
    let mut h = [0f64; 9];
    let mut inliers = 0;
    let rc = unsafe {
        sc_fit_homography(
            src_flat.as_ptr(),
            dst_flat.as_ptr(),
            src.len() as i32,
            ransac,
            h.as_mut_ptr(),
            &mut inliers,
        )
    };
    if rc != 0 {
        return Err("圆点单应拟合失败".into());
    }
    Ok((h, inliers))
}

fn save_overlap(src: i32, dst: i32, h: &Mat3, path: &Path) -> Result<(), String> {
    let cpath = cstring(path)?;
    let mut err = [0 as c_char; 512];
    let rc = unsafe {
        sc_save_overlap(
            src,
            dst,
            h.as_ptr(),
            cpath.as_ptr(),
            err.as_mut_ptr(),
            err.len() as i32,
        )
    };
    if rc != 0 {
        return Err(format!("保存重叠图失败: {}", cerr(&err)));
    }
    Ok(())
}

fn stitch(handles: &[i32], h_to_ref: &[f64], path: &Path) -> Result<ScCanvas, String> {
    let cpath = cstring(path)?;
    let mut canvas = ScCanvas {
        min_x: 0.0,
        min_y: 0.0,
        fit: 1.0,
        width_before_rot: 0,
        height_before_rot: 0,
        out_width: 0,
        out_height: 0,
    };
    let mut err = [0 as c_char; 512];
    let rc = unsafe {
        sc_stitch(
            handles.as_ptr(),
            h_to_ref.as_ptr(),
            handles.len() as i32,
            cpath.as_ptr(),
            1,
            &mut canvas,
            err.as_mut_ptr(),
            err.len() as i32,
        )
    };
    if rc != 0 {
        return Err(format!("拼接失败: {}", cerr(&err)));
    }
    Ok(canvas)
}

fn flatten(pts: &[[f64; 2]]) -> Vec<f64> {
    let mut v = Vec::with_capacity(pts.len() * 2);
    for p in pts {
        v.push(p[0]);
        v.push(p[1]);
    }
    v
}

fn cstring(path: &Path) -> Result<CString, String> {
    let text = path.to_string_lossy();
    CString::new(text.as_ref()).map_err(|_| format!("路径含空字符: {}", path.display()))
}

fn cerr(buf: &[c_char]) -> String {
    unsafe {
        CStr::from_ptr(buf.as_ptr())
            .to_string_lossy()
            .into_owned()
    }
}

fn write_calibration(
    path: &Path,
    shots: &[Shot],
    pairs: &[PairCalib],
    h_to_1: &HashMap<u32, Mat3>,
    primary: &HashMap<u32, usize>,
    scale: f64,
    sensor_w: i32,
    sensor_h: i32,
    canvas: &ScCanvas,
) -> Result<(), String> {
    let mut json = String::new();
    json.push_str("{\n");
    json.push_str(&format!(
        "  \"sensor\": {{\"width\": {sensor_w}, \"height\": {sensor_h}}},\n"
    ));
    json.push_str(&format!("  \"working_scale\": {scale:.4},\n"));
    json.push_str("  \"reference_camera\": 1,\n");
    json.push_str("  \"camera_order_left_to_right\": [1, 2, 3, 4, 5, 6],\n");
    json.push_str("  \"grid\": {\"rows\": 7, \"cols\": 7, \"type\": \"symmetric_circles\", \"spacing\": 1.0, \"spacing_unit\": \"grid_step\"},\n");
    json.push_str("  \"geometry\": \"相机2与其余相机光轴夹角约90度。标定板和图纸都在同一桌面上，相邻相机用平面单应性连接，再链式投到相机1。\",\n");
    json.push_str("  \"H_convention\": \"H 作用于齐次像素 [x, y, 1]，结果在相机1的全分辨率像素坐标系。working_scale 下的单应已换算回全分辨率。\",\n");
    json.push_str("  \"display\": {\n");
    json.push_str("    \"rotation\": \"ROTATE_90_COUNTERCLOCKWISE\",\n");
    json.push_str("    \"reason\": \"相机1像素的 +Y 指向相机6。逆时针转90度后，相机1在左、相机6在右。\",\n");
    json.push_str(&format!(
        "    \"canvas_before_rotation\": [{}, {}],\n",
        canvas.width_before_rot, canvas.height_before_rot
    ));
    json.push_str(&format!(
        "    \"origin_xy_in_camera1_work\": [{:.4}, {:.4}],\n",
        canvas.min_x, canvas.min_y
    ));
    json.push_str(&format!("    \"fit\": {:.8},\n", canvas.fit));
    json.push_str(&format!(
        "    \"mosaic_size\": [{}, {}]\n",
        canvas.out_width, canvas.out_height
    ));
    json.push_str("  },\n");
    json.push_str("  \"pairs\": [\n");
    for (i, pair) in pairs.iter().enumerate() {
        let comma = if i + 1 == pairs.len() { "" } else { "," };
        json.push_str("    {\n");
        json.push_str(&format!("      \"src\": {},\n", pair.src));
        json.push_str(&format!("      \"dst\": {},\n", pair.dst));
        json.push_str(&format!("      \"method\": \"{}\",\n", pair.method));
        json.push_str(&format!("      \"sift_matches\": {},\n", pair.sift_matches));
        json.push_str(&format!("      \"sift_inliers\": {},\n", pair.sift_inliers));
        json.push_str(&format!("      \"circle_points\": {},\n", pair.circle_points));
        json.push_str(&format!(
            "      \"circle_rms_px_work\": {},\n",
            opt_num(pair.circle_rms_px)
        ));
        json.push_str(&format!(
            "      \"board_vs_sift_median_px_work\": {},\n",
            opt_num(pair.board_sift_median_px)
        ));
        json.push_str(&format!(
            "      \"H_full_src_to_dst\": {}\n",
            mat_json(&to_full(&pair.h_work, scale))
        ));
        json.push_str(&format!("    }}{comma}\n"));
    }
    json.push_str("  ],\n");
    json.push_str("  \"H_full_to_camera1\": {\n");
    for cam in 1u32..=6 {
        let comma = if cam == 6 { "" } else { "," };
        json.push_str(&format!(
            "    \"{cam}\": {}{comma}\n",
            mat_json(&to_full(&h_to_1[&cam], scale))
        ));
    }
    json.push_str("  },\n");
    json.push_str("  \"primary_image\": {\n");
    for cam in 1u32..=6 {
        let comma = if cam == 6 { "" } else { "," };
        json.push_str(&format!(
            "    \"{cam}\": \"{}\"{comma}\n",
            json_escape(&shots[primary[&cam]].stem)
        ));
    }
    json.push_str("  },\n");
    json.push_str("  \"images\": [\n");
    for (i, shot) in shots.iter().enumerate() {
        let comma = if i + 1 == shots.len() { "" } else { "," };
        json.push_str("    {\n");
        json.push_str(&format!("      \"file\": \"{}\",\n", json_escape(&shot.path.display().to_string())));
        json.push_str(&format!("      \"folder\": \"{}\",\n", json_escape(&shot.folder)));
        json.push_str(&format!("      \"camera\": {},\n", shot.camera));
        json.push_str(&format!("      \"grid_count\": {},\n", shot.grid.len()));
        json.push_str("      \"grid_xy_full\": [");
        for (k, p) in shot.grid.iter().enumerate() {
            if k > 0 {
                json.push_str(", ");
            }
            json.push_str(&format!("[{:.3}, {:.3}]", p[0] / scale, p[1] / scale));
        }
        json.push_str("]\n");
        json.push_str(&format!("    }}{comma}\n"));
    }
    json.push_str("  ]\n");
    json.push_str("}\n");
    fs::write(path, json).map_err(|e| e.to_string())
}

fn opt_num(v: Option<f64>) -> String {
    match v {
        Some(n) => format!("{n:.4}"),
        None => "null".into(),
    }
}

fn mat_json(h: &Mat3) -> String {
    let parts: Vec<String> = h.iter().map(|v| format!("{v:.8}")).collect();
    format!("[{}]", parts.join(", "))
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}
