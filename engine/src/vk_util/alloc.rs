use std::ptr::NonNull;
use std::ffi;

use erupt::{DeviceLoader, vk};
use anyhow::{Result, Context, bail};

use super::PhysicalDeviceInfo;

const MIN_DEVICE_VK_MEM_SIZE: u64 = 16 * 1024 * 1024; // 16 MiB
const MIN_HOST_VK_MEM_SIZE: u64 = 4 * 1024 * 1024; // 4 MiB

fn padding(addr: u64, alignment: u64) -> u64 {
    if alignment != 0 {
        (alignment - addr % alignment) % alignment        
    }
    else {
        0
    }
}

pub struct MemoryRegion {
    offset: u64,
    size: u64
}

#[derive(Clone, Copy, Debug)]
pub enum MemoryType {
    Device,
    Host
}

struct BlockSource {
    mem_type: MemoryType,
    idx: usize
}

pub struct MemoryBlock {
    mem: vk::DeviceMemory,
    region: MemoryRegion,
    ptr: Option<NonNull<ffi::c_void>>,
    source: BlockSource
}

impl MemoryBlock {
    pub fn mem(&self) -> vk::DeviceMemory {
        self.mem
    }

    pub fn ptr(&self) -> Option<NonNull<ffi::c_void>> {
        self.ptr
    }

    pub fn offset(&self) -> u64 {
        self.region.offset
    }
}

struct VkMemory {
    mem: vk::DeviceMemory,
    free_regions: Vec<MemoryRegion>,
    base_ptr: Option<NonNull<ffi::c_void>>
}

impl VkMemory {
    fn alloc(
        &mut self,
        req: &vk::MemoryRequirements,
        mem_type: MemoryType,
        idx: usize
    ) -> Option<MemoryBlock> {
        // Find a free region that can fit our allocation
        let mut block = None;
        let mut removal_idx = None;

        for (i, region) in self.free_regions.iter_mut().enumerate() {
            let padding = padding(region.offset, req.alignment);
            let padded_size = req.size + padding;

            if padded_size <= region.size {
                // Our allocation can fit in this free region
                // Allocate here
                let ptr = self.base_ptr.map(|base_ptr| unsafe {
                    let ptr = base_ptr
                        .as_ptr()
                        .offset((region.offset + padding) as isize);

                    NonNull::new(ptr).unwrap()
                });

                block = Some(MemoryBlock {
                    mem: self.mem,
                    region: MemoryRegion { offset: region.offset + padding, size: padded_size },
                    ptr,
                    source: BlockSource { mem_type, idx }
                });

                // If our allocation fits exactly into this free region, remove it
                // Otherwise shrink it
                if padded_size == region.size {
                    removal_idx = Some(i);
                }
                else {
                    region.offset += padded_size;
                    region.size -= padded_size;
                }                

                break;
            }
        }

        // Remove free region if needed
        if let Some(idx) = removal_idx {
            self.free_regions.remove(idx);
        }

        block
    }

    fn free(&mut self, returning_region: MemoryRegion) {
        let end_addr = returning_region.offset + returning_region.size;

        // Find first free region thats located after the returning region
        let following_region = self.free_regions
            .iter_mut()
            .enumerate()
            .find(|(_i, free_region)| free_region.offset >= end_addr);

        match following_region {
            // Following region found
            Some((i, following_region)) => {
                // If the following region is contigious with the returning region
                // merge it with the following region
                // Else insert the returning region just before it
                if end_addr == following_region.offset {
                    following_region.offset -= returning_region.size;
                    following_region.size += returning_region.size;
                }
                else {
                    self.free_regions.insert(i, returning_region);
                }
            },

            // No following region, place returning region at the end
            None => self.free_regions.push(returning_region)
        }
    }
}

struct VkMemoryManager {
    mem_type: MemoryType,
    mem_type_idx: u32,
    should_map: bool,
    min_vk_mem_size: u64,
    vk_mems: Vec<VkMemory>
}

impl VkMemoryManager {
    fn alloc(&mut self, device: &DeviceLoader, req: &vk::MemoryRequirements) -> Result<MemoryBlock> {
        // Try allocating in each VkMemory
        let block = self.vk_mems
            .iter_mut()
            .enumerate()
            .find_map(|(idx, vk_mem)| vk_mem.alloc(req, self.mem_type, idx));

        match block {
            // Allocation done
            Some(block) => Ok(block),

            // Failed to allocate, create new VkMemory to allocate from
            None => {
                // Respect minimum VkMemory size
                let alloc_size = req.size.max(self.min_vk_mem_size);
                
                let alloc_info = vk::MemoryAllocateInfoBuilder::new()
                    .allocation_size(alloc_size)
                    .memory_type_index(self.mem_type_idx);

                let mem = unsafe { device.allocate_memory(&alloc_info, None) }
                    .result()
                    .context("Failed to allocate VkMemory")?;

                let base_ptr = if self.should_map {
                    let ptr = unsafe { device.map_memory(mem, 0, alloc_size, vk::MemoryMapFlags::empty()) }
                        .result()
                        .context("Failed to map memory")?;

                    Some(NonNull::new(ptr).unwrap())
                }
                else {
                    None
                };

                let block = MemoryBlock {
                    mem,
                    region: MemoryRegion { offset: 0, size: req.size },
                    ptr: base_ptr,
                    source: BlockSource { mem_type: self.mem_type, idx: self.vk_mems.len() }
                };

                // When req.size >= min_vk_mem_size, the entire VkMemory is dedicated
                // to this one allocation
                // There are no free regions in that case, otherwise there will be a free region
                let free_regions = if req.size >= self.min_vk_mem_size {
                    vec![]
                }
                else {
                    vec![MemoryRegion { offset: req.size, size: self.min_vk_mem_size - req.size }]
                };

                // Add the new VkMemory
                self.vk_mems.push(VkMemory { mem, free_regions, base_ptr });

                Ok(block)
            }
        }
    }

    fn free(&mut self, block: MemoryBlock) {
        self.vk_mems[block.source.idx].free(block.region)
    }

    fn destroy(self, device: &DeviceLoader) {
        for vk_mem in self.vk_mems {
            unsafe { device.free_memory(vk_mem.mem, None); }
        }
    }
}

pub struct VkAllocator {
    device_mgr: VkMemoryManager,
    host_mgr: VkMemoryManager
}

impl VkAllocator {
    pub fn new(phys_dev_info: &PhysicalDeviceInfo) -> Result<Self> {
        let mem_props = phys_dev_info.mem_props();
        let mem_types = &mem_props.memory_types[..mem_props.memory_type_count as usize];

        // Find memory type indices
        let find_memory = |props| {
            mem_types
                .iter()
                .enumerate()
                .find_map(|(i, mem_type)| (mem_type.property_flags == props).then_some(i as u32))
        };

        let device_mem_type_idx = find_memory(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .context("Failed to find device memory type")?;

        let host_mem_type_idx = find_memory(
            vk::MemoryPropertyFlags::HOST_VISIBLE |
            vk::MemoryPropertyFlags::HOST_CACHED |
            vk::MemoryPropertyFlags::HOST_COHERENT
        ).context("Failed to find host memory type")?;

        // Create the memory type managers
        let device_mgr = VkMemoryManager {
            mem_type: MemoryType::Device,
            mem_type_idx: device_mem_type_idx,
            should_map: false,
            min_vk_mem_size: MIN_DEVICE_VK_MEM_SIZE,
            vk_mems: vec![]
        };

        let host_mgr = VkMemoryManager {
            mem_type: MemoryType::Host,
            mem_type_idx: host_mem_type_idx,
            should_map: true,
            min_vk_mem_size: MIN_HOST_VK_MEM_SIZE,
            vk_mems: vec![]
        };

        Ok(Self {
            device_mgr,
            host_mgr
        })
    }

    pub fn alloc(
        &mut self,
        device: &DeviceLoader,
        req: &vk::MemoryRequirements,
        mem_type: MemoryType
    ) -> Result<MemoryBlock> {
        let mgr = match mem_type {
            MemoryType::Device => &mut self.device_mgr,
            MemoryType::Host => &mut self.host_mgr
        };

        // Check if the allocation is valid for the memory type
        if req.memory_type_bits & (1 << mgr.mem_type_idx) == 0 {
            bail!("This allocation cannot be done in {mem_type:?} memory");
        }

        // Try and allocate
        mgr.alloc(device, req)
    }

    pub fn free(&mut self, block: MemoryBlock) {
        let mgr = match block.source.mem_type {
            MemoryType::Device => &mut self.device_mgr,
            MemoryType::Host => &mut self.host_mgr,
        };

        mgr.free(block)
    }

    pub fn destroy(self, device: &DeviceLoader) {
        self.device_mgr.destroy(device);
        self.host_mgr.destroy(device);
    }
}