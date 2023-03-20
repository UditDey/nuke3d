//! 2D vector graphics system

mod canvas;

use std::mem;
use std::slice;
use std::ffi::CString;

use ash::{vk, Device};
use anyhow::{Result, Context};

use super::vk_util::{
    buffer::TransferBuffer,
    vma::VmaAllocator,
    frame_queue::{FrameQueue, FrameInfo}
};

use canvas::Canvas2D;

const BOUNDARY_ELEM_BUF_SIZE: u64 = 8 * 1024; // 8 KiB

// Compute shader specialization constants
const WORKGROUP_SIZE: u32 = 8;
const NUM_SAMPLES: u32 = 5;

#[repr(C)]
struct Metadata {
    num_lines: u32
}

pub struct Canvas2DSystem {
    lines_bufs: Vec<TransferBuffer>,
    desc_pool: vk::DescriptorPool,
    buf_desc_sets: Vec<vk::DescriptorSet>,
    image_desc_sets: Vec<vk::DescriptorSet>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline
}

impl Canvas2DSystem {
    pub fn new(
        device: &Device,
        frame_queue: &FrameQueue,
        vma_alloc: &VmaAllocator,
        frames_in_flight: u32
    ) -> Result<Self> {
        // Create boundary element buffers
        let create_info = vk::BufferCreateInfo::builder()
            .size(BOUNDARY_ELEM_BUF_SIZE)
            .usage(vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
            
        let lines_bufs = (0..frames_in_flight)
            .map(|_| TransferBuffer::new(vma_alloc, &create_info))
            .collect::<Result<Vec<TransferBuffer>>>()
            .context("Failed to create lines transfer buffers")?;
            
        // Create element buffer descriptor set layout
        let bindings = [
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build()
        ];
        
        let create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        
        let buf_set_layout = unsafe { device.create_descriptor_set_layout(&create_info, None) }
            .context("Failed to create element buffer descriptor set layout")?;
            
        // Create swapchain image descriptor set layout
        let bindings = [
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build()
        ];
        
        let create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        
        let image_set_layout = unsafe { device.create_descriptor_set_layout(&create_info, None) }
            .context("Failed to create swapchain image descriptor set layout")?;
        
        // Create descriptor pool
        // Number of UNIFORM_BUFFER descriptors = 1 * frames_in_flight
        // Number of STORAGE_IMAGE descriptors = number of swapchain images
        // Number of descriptor sets = frames_in_flight + number of swapchain images
        let num_swap_images = frame_queue.swap_image_views().len() as u32;
        
        let pool_sizes = [
            vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1 * frames_in_flight)
                .build(),

            vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(num_swap_images)
                .build()
        ];
        
        let create_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(frames_in_flight + num_swap_images)
            .pool_sizes(&pool_sizes);

        let desc_pool = unsafe { device.create_descriptor_pool(&create_info, None) }
            .context("Failed to create descriptor pool")?;
            
        // Allocate element buffer descriptor sets
        let set_layouts = vec![buf_set_layout; frames_in_flight as usize];

        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(desc_pool)
            .set_layouts(&set_layouts);

        let buf_desc_sets = unsafe { device.allocate_descriptor_sets(&alloc_info) }
            .context("Failed to allocate element buffer descriptor sets")?;
            
        // Allocate swapchain image descriptor sets
        let set_layouts = vec![image_set_layout; num_swap_images as usize];

        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(desc_pool)
            .set_layouts(&set_layouts);

        let image_desc_sets = unsafe { device.allocate_descriptor_sets(&alloc_info) }
            .context("Failed to allocate swapchain image descriptor sets")?;
            
        // Update element buffer descriptor sets
        let lines_bufs_infos = lines_bufs
            .iter()
            .map(|buf| {
                let info = vk::DescriptorBufferInfo {
                    buffer: buf.buf(),
                    offset: 0,
                    range: buf.size()
                };
                
                [info]
            })
            .collect::<Vec<_>>();
            
        // let quadratics_bufs_infos = ...
            
        let lines_bufs_writes = buf_desc_sets
            .iter()
            .zip(&lines_bufs_infos)
            .map(|(desc_set, buf_info)| {
                vk::WriteDescriptorSet::builder()
                    .dst_set(*desc_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(buf_info)
                    .build()
            });
            
        // let quadratics_bufs_write = ...
            
        let writes = lines_bufs_writes.collect::<Vec<vk::WriteDescriptorSet>>();
        // let writes = lines_bufs_writes.chain(quadratics_bufs_write) ...
        
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        
        // Update swapchain image descriptor sets
        let swap_image_infos = frame_queue
            .swap_image_views()
            .iter()
            .map(|&image_view| {
                let info = vk::DescriptorImageInfo {
                    sampler: vk::Sampler::null(),
                    image_view,
                    image_layout: vk::ImageLayout::GENERAL
                };
                
                [info]
            })
            .collect::<Vec<_>>();
            
        let writes = image_desc_sets
            .iter()
            .zip(&swap_image_infos)
            .map(|(desc_set, image_info)| {
                vk::WriteDescriptorSet::builder()
                    .dst_set(*desc_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(image_info)
                    .build()
            })
            .collect::<Vec<vk::WriteDescriptorSet>>();
            
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        
        // Create pipeline layout
        let set_layouts = [buf_set_layout, image_set_layout];

        let push_const_ranges = [
            vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(mem::size_of::<Metadata>() as u32)
                .build()
        ];
        
        let create_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_const_ranges);

        let pipeline_layout = unsafe { device.create_pipeline_layout(&create_info, None) }
            .context("Failed to create pipeline layout")?;
            
        // Create shader module
        let shader_spv = include_bytes!(concat!(
            "..", env!("PATH_SEPERATOR"),
            "..", env!("PATH_SEPERATOR"),
            "..", env!("PATH_SEPERATOR"),
            "shaders", env!("PATH_SEPERATOR"),
            "canvas_2d.spv"
        )).as_slice();

        // Convert [u8] to [u32]
        let shader_spv = unsafe {
            let len = shader_spv.len() / 4;
            slice::from_raw_parts(shader_spv.as_ptr() as *const u32, len)
        };

        let create_info = vk::ShaderModuleCreateInfo::builder().code(shader_spv);

        let shader_module = unsafe { device.create_shader_module(&create_info, None) }
            .context("Failed to create canvas2d shader module")?;
            
        // Create compute pipeline
        let spec_consts_buf = [WORKGROUP_SIZE, NUM_SAMPLES, BOUNDARY_ELEM_BUF_SIZE as u32].as_slice();

        // Convert [u32] to [u8]
        let spec_consts_buf = unsafe {
            let len = shader_spv.len() * 4;
            slice::from_raw_parts(spec_consts_buf.as_ptr() as *const u8, len)
        };

        let spec_consts_entries = [
            // Workgroup size
            vk::SpecializationMapEntry::builder()
                .constant_id(0)
                .offset(0)
                .size(4)
                .build(),

            // Num samples
            vk::SpecializationMapEntry::builder()
                .constant_id(1)
                .offset(4)
                .size(4)
                .build(),
                
            // Boundary element buffer size
            vk::SpecializationMapEntry::builder()
                .constant_id(2)
                .offset(8)
                .size(4)
                .build()
        ];

        let specialization_info = vk::SpecializationInfo::builder()
            .map_entries(&spec_consts_entries)
            .data(spec_consts_buf);

        let entry_point = CString::new("main").unwrap();

        let stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry_point)
            .specialization_info(&specialization_info)
            .build();

        let create_info = vk::ComputePipelineCreateInfo::builder()
            .stage(stage_create_info)
            .layout(pipeline_layout)
            .build();

        let pipeline = unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None) }
            .map_err(|(_, result)| result)
            .context("Failed to create canvas compute pipeline")?[0];
        
        // Destroy unneeded objects
        unsafe {
            device.destroy_shader_module(shader_module, None);
            device.destroy_descriptor_set_layout(buf_set_layout, None);
            device.destroy_descriptor_set_layout(image_set_layout, None);
        }
        
        Ok(Self {
            lines_bufs,
            desc_pool,
            buf_desc_sets,
            image_desc_sets,
            pipeline_layout,
            pipeline
        })
    }
    
    /// Create a new [`Canvas2D`] for a frame
    pub fn new_canvas(&self, frame_info: &FrameInfo) -> Canvas2D {
        Canvas2D::new(&self.lines_bufs[frame_info.frame_idx()])
    }
    
    pub fn draw(&self, device: &Device, cmd_buf: vk::CommandBuffer, frame_info: &FrameInfo, canvas_2d: Canvas2D) {
        unsafe {
            // Bind pipeline
            device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            
            // Bind relevant descriptor sets
            let sets = [self.buf_desc_sets[frame_info.frame_idx()], self.image_desc_sets[frame_info.swap_image_idx()]];

            device.cmd_bind_descriptor_sets(
                cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &sets,
                &[]
            );
            
            // Push metadata
            let ptr = &canvas_2d.metadata as *const Metadata as *const u8;
            let data = slice::from_raw_parts(ptr, mem::size_of::<Metadata>());

            device.cmd_push_constants(cmd_buf, self.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, data);
            
            let workgroups_x = (frame_info.swap_image_extent().width + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            let workgroups_y = (frame_info.swap_image_extent().height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            
            device.cmd_dispatch(cmd_buf, workgroups_x, workgroups_y, 1);
        }
    }
    
    pub fn destroy(self, device: &Device, vma_alloc: &VmaAllocator) {
        for buf in self.lines_bufs {
            buf.destroy(vma_alloc);
        }

        unsafe {
            device.destroy_descriptor_pool(self.desc_pool, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_pipeline(self.pipeline, None);
        }
    }
}