fn main() {
    // Build VMA
    let mut build = cc::Build::new();

    build.include("ffi_libs/VulkanMemoryAllocator/include");
}