use std::env;
use std::path::{MAIN_SEPARATOR, Path, PathBuf};

use bindgen::callbacks::{ParseCallbacks, DeriveInfo};

fn main() {
    // Set a PATH_SEPERATOR rustc variable to make include!() cross platform
    println!("cargo:rustc-env=PATH_SEPERATOR={}", MAIN_SEPARATOR);

    // Build VMA
    let mut build = cc::Build::new();

    // Includes
    let vma_include_path = Path::new("ffi_libs").join("VulkanMemoryAllocator").join("include");
    let vk_header_include_path = Path::new("ffi_libs").join("Vulkan-Headers").join("include");

    build.include(&vma_include_path);
    build.include(&vk_header_include_path);

    // Defines
    build.define("NDEBUG", "");
    build.define("VMA_STATIC_VULKAN_FUNCTIONS", "0");
    build.define("VMA_DYNAMIC_VULKAN_FUNCTIONS", "0");
    build.define("VMA_STATS_STRING_ENABLED", "0");
    build.define("VMA_IMPLEMENTATION", "");

    // cpp files
    build.file(Path::new("ffi_libs").join("vma.cpp"));

    // Build it
    build
        .cpp(true)
        .flag("-std=c++17")
        .flag("-Wno-missing-field-initializers")
        .flag("-Wno-unused-variable")
        .flag("-Wno-unused-parameter")
        .flag("-Wno-unused-private-field")
        .flag("-Wno-reorder")
        .flag("-Wno-parentheses")
        .flag("-Wno-implicit-fallthrough")
        .cpp_link_stdlib("stdc++")
        .debug(true) // Always emit debug info
        .compile("vma");

    // Generate FFI bindings
    let bindings = bindgen::Builder::default()
        .clang_arg(format!("-I{}", vk_header_include_path.to_string_lossy()))
        .header(vma_include_path.join("vk_mem_alloc.h").to_string_lossy())
        .rustfmt_bindings(true)
        .size_t_is_usize(true)
        .blocklist_type("__darwin_.*")
        .allowlist_function("vma.*")
        .allowlist_function("PFN_vma.*")
        .allowlist_type("Vma.*")
        .blocklist_type("Vk.*")
        .blocklist_type("PFN_vk.*")
        .trust_clang_mangling(false)
        .layout_tests(false)
        .rustified_enum("Vma.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .parse_callbacks(Box::new(Fixes))
        .generate()
        .unwrap();

    let binding_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings.write_to_file(binding_path.join("vma_ffi.rs")).unwrap();
}

#[derive(Debug)]
struct Fixes;

impl ParseCallbacks for Fixes {
    fn item_name(&self, original_item_name: &str) -> Option<String> {
        if original_item_name.starts_with("Vk") {
            Some(original_item_name.trim_start_matches("Vk").to_string())
        } else if original_item_name.starts_with("PFN_vk") && original_item_name.ends_with("KHR") {
            Some(original_item_name.trim_end_matches("KHR").to_string())
        } else {
            None
        }
    }

    fn add_derives(&self, info: &DeriveInfo) -> Vec<String> {
        if info.name.starts_with("VmaAllocationInfo") || info.name.starts_with("VmaDefragmentationStats") {
            vec!["Debug".into(), "Copy".into(), "Clone".into()]
        } else {
            vec![]
        }
    }
}