//! Bindings to AMD's Vulkan Memory Allocator

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::ffi::c_void;
use std::ptr::{self, NonNull};
use std::mem::{self, MaybeUninit};

use ash::{vk, Instance, Device};
use anyhow::{Result, Context};

mod ffi {
    use ash::vk::*;
    include!(concat!(env!("OUT_DIR"), env!("PATH_SEPERATOR"), "/vma_ffi.rs"));
}

fn null_vk_fn() {
    panic!("Null VMA vulkan function called")
}

macro_rules! null_vk_fn {
    () => {
        unsafe { mem::transmute(null_vk_fn as usize) }
    };
}

/// Memory allocation types
pub enum AllocType {
    Host,
    Device
}

/// A vulkan buffer with memory allocated and bound to it
pub struct BufferAlloc {
    buf: vk::Buffer,
    ptr: Option<NonNull<c_void>>,
    alloc: ffi::VmaAllocation
}

impl BufferAlloc {
    /// The underlying [`vk::Buffer`]
    pub fn buf(&self) -> vk::Buffer {
        self.buf
    }
}

/// Vulkan Memory Allocator
pub struct VmaAllocator {
    vma_alloc: ffi::VmaAllocator
}

impl VmaAllocator {
    pub fn new(instance: &Instance, phys_dev: vk::PhysicalDevice, device: &Device) -> Result<Self> {
        // Create allocator
        let vk_fns = ffi::VmaVulkanFunctions {
            vkGetInstanceProcAddr: null_vk_fn!(),
            vkGetDeviceProcAddr: null_vk_fn!(),
            vkGetPhysicalDeviceProperties: instance.fp_v1_0().get_physical_device_properties,
            vkGetPhysicalDeviceMemoryProperties: instance.fp_v1_0().get_physical_device_memory_properties,
            vkAllocateMemory: device.fp_v1_0().allocate_memory,
            vkFreeMemory: device.fp_v1_0().free_memory,
            vkMapMemory: device.fp_v1_0().map_memory,
            vkUnmapMemory: device.fp_v1_0().unmap_memory,
            vkFlushMappedMemoryRanges: device.fp_v1_0().flush_mapped_memory_ranges,
            vkInvalidateMappedMemoryRanges: device.fp_v1_0().invalidate_mapped_memory_ranges,
            vkBindBufferMemory: device.fp_v1_0().bind_buffer_memory,
            vkBindImageMemory: device.fp_v1_0().bind_image_memory,
            vkGetBufferMemoryRequirements: device.fp_v1_0().get_buffer_memory_requirements,
            vkGetImageMemoryRequirements: device.fp_v1_0().get_image_memory_requirements,
            vkCreateBuffer: device.fp_v1_0().create_buffer,
            vkDestroyBuffer: device.fp_v1_0().destroy_buffer,
            vkCreateImage: device.fp_v1_0().create_image,
            vkDestroyImage: device.fp_v1_0().destroy_image,
            vkCmdCopyBuffer: device.fp_v1_0().cmd_copy_buffer,
            vkGetBufferMemoryRequirements2KHR: null_vk_fn!(),
            vkGetImageMemoryRequirements2KHR: null_vk_fn!(),
            vkBindBufferMemory2KHR: null_vk_fn!(),
            vkBindImageMemory2KHR: null_vk_fn!(),
            vkGetPhysicalDeviceMemoryProperties2KHR: null_vk_fn!(),
            vkGetDeviceBufferMemoryRequirements: null_vk_fn!(),
            vkGetDeviceImageMemoryRequirements: null_vk_fn!()
        };

        let create_info = ffi::VmaAllocatorCreateInfo {
            flags: ffi::VmaAllocatorCreateFlagBits::VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT as u32,
            physicalDevice: phys_dev,
            device: device.handle(),
            preferredLargeHeapBlockSize: 0,
            pAllocationCallbacks: ptr::null(),
            pDeviceMemoryCallbacks: ptr::null(),
            pHeapSizeLimit: ptr::null(),
            pVulkanFunctions: &vk_fns,
            instance: instance.handle(),
            vulkanApiVersion: vk::make_api_version(0, 1, 0, 0),
            pTypeExternalMemoryHandleTypes: ptr::null()
        };

        let vma_alloc = unsafe {
            let mut vma_alloc = MaybeUninit::uninit();

            ffi::vmaCreateAllocator(&create_info, vma_alloc.as_mut_ptr())
                .result()
                .context("Failed to create VMA allocator")?;

            vma_alloc.assume_init()
        };

        Ok(Self { vma_alloc })
    }

    pub fn create_buffer(&self, create_info: &vk::BufferCreateInfo, alloc_type: AllocType) -> Result<BufferAlloc> {
        let mut flags = 0;

        if let AllocType::Host = alloc_type {
            flags |= ffi::VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT as u32
        }

        let usage = match alloc_type {
            AllocType::Host => ffi::VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
            AllocType::Device => ffi::VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
        };

        let alloc_info = ffi::VmaAllocationCreateInfo {
            flags,
            usage,
            requiredFlags: vk::MemoryPropertyFlags::empty(),
            preferredFlags: vk::MemoryPropertyFlags::empty(),
            memoryTypeBits: 0,
            pool: unsafe { mem::zeroed() },
            pUserData: ptr::null_mut(),
            priority: 1.0
        };

        let buf_alloc = unsafe {
            let mut buf = MaybeUninit::uninit();
            let mut allocation = MaybeUninit::uninit();
            let mut allocation_info = MaybeUninit::uninit();

            ffi::vmaCreateBuffer(
                self.vma_alloc,
                create_info,
                &alloc_info,
                buf.as_mut_ptr(),
                allocation.as_mut_ptr(),
                allocation_info.as_mut_ptr()
            )
            .result()
            .context("vmaCreateBuffer failed")?;

            let buf = buf.assume_init();
            let allocation = allocation.assume_init();
            let allocation_info = allocation_info.assume_init();

            let ptr = NonNull::new(allocation_info.pMappedData as *mut c_void);

            BufferAlloc {
                buf,
                ptr,
                alloc: allocation
            }
        };

        Ok(buf_alloc)
    }
}

impl Drop for VmaAllocator {
    fn drop(&mut self) {
        unsafe { ffi::vmaDestroyAllocator(self.vma_alloc); }
    }
}