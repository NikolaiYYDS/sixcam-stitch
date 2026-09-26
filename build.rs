use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let cpp = manifest.join("cpp");
    // CMake 4 crashes when the binary dir sits on a non-ASCII path.
    let out = PathBuf::from(r"D:\tmp\sixcam-stitch-cmake");
    std::fs::create_dir_all(&out).expect("create cmake build dir");
    let opencv_dir = std::env::var("OpenCV_DIR").unwrap_or_else(|_| r"C:\opencv\build".to_string());

    println!("cargo:rerun-if-changed=cpp/CMakeLists.txt");
    println!("cargo:rerun-if-changed=cpp/include/stitch_core.h");
    println!("cargo:rerun-if-changed=cpp/src/stitch_core.cpp");

    let configure = Command::new("cmake")
        .arg("-S")
        .arg(&cpp)
        .arg("-B")
        .arg(&out)
        .arg(format!("-DOpenCV_DIR={opencv_dir}"))
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .output()
        .expect("cmake not found on PATH");
    if !configure.status.success() {
        eprintln!("{}", String::from_utf8_lossy(&configure.stdout));
        eprintln!("{}", String::from_utf8_lossy(&configure.stderr));
        panic!("cmake configure failed");
    }

    let built = Command::new("cmake")
        .arg("--build")
        .arg(&out)
        .arg("--config")
        .arg("Release")
        .output()
        .expect("cmake build failed to start");
    if !built.status.success() {
        eprintln!("{}", String::from_utf8_lossy(&built.stdout));
        eprintln!("{}", String::from_utf8_lossy(&built.stderr));
        panic!("cmake build failed");
    }

    let lib_dir = find_artifact(&out, "stitch_core.lib").expect("stitch_core.lib not found");
    let dll = find_artifact(&out, "stitch_core.dll").expect("stitch_core.dll not found");
    println!("cargo:rustc-link-search=native={}", lib_dir.parent().unwrap().display());
    println!("cargo:rustc-link-lib=dylib=stitch_core");

    let profile_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap())
        .join("..")
        .join("..")
        .join("..");
    let profile_dir = profile_dir.canonicalize().expect("target profile dir");
    std::fs::copy(&dll, profile_dir.join("stitch_core.dll")).expect("copy stitch_core.dll");

    let opencv_bin = PathBuf::from(&opencv_dir).join("x64").join("vc16").join("bin");
    let world = opencv_bin.join("opencv_world4110.dll");
    if world.exists() {
        std::fs::copy(&world, profile_dir.join("opencv_world4110.dll")).expect("copy opencv dll");
    } else {
        println!("cargo:warning=opencv_world4110.dll not found at {}", world.display());
    }
}

fn find_artifact(root: &Path, name: &str) -> Option<PathBuf> {
    let direct = [root.join(name), root.join("Release").join(name), root.join("Debug").join(name)];
    for path in &direct {
        if path.exists() {
            return Some(path.clone());
        }
    }
    for entry in std::fs::read_dir(root).ok()?.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_artifact(&path, name) {
                return Some(found);
            }
        }
    }
    None
}
