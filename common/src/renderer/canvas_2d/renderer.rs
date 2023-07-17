use std::slice;
use std::ffi::CString;

use ash::{vk, Device};
use anyhow::{Result, Context};

use crate::renderer::vk_util::{
    frame_queue::{FrameQueue, FrameInfo},
    vma::VmaAllocator,
    buffer::TransferBuffer
};

use super::recorder::{CMD_LIST_BUF_SIZE, Canvas2DRecorder, InitState};

const WG_SIZE: u32 = 8; // Workgroup size = (8, 8)

/// Canvas2D renderer
pub struct Canvas2DRenderer {
    cmd_list_bufs: Vec<TransferBuffer>,
    desc_pool: vk::DescriptorPool,
    cmd_list_desc_sets: Vec<vk::DescriptorSet>,
    image_desc_sets: Vec<vk::DescriptorSet>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline
}

impl Canvas2DRenderer {
    pub fn new(
        device: &Device,
        frame_queue: &FrameQueue,
        vma_alloc: &VmaAllocator,
        frames_in_flight: u32
    ) -> Result<Self> {
        // Create canvas command list buffers
        let cmd_list_bufs = {
            let create_info = vk::BufferCreateInfo::builder()
                .size(CMD_LIST_BUF_SIZE)
                .usage(vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
                
            (0..frames_in_flight)
                .map(|_| TransferBuffer::new(vma_alloc, &create_info))
                .collect::<Result<Vec<TransferBuffer>>>()
                .context("Failed to create canvas command list buffers")?
        };
            
        // Create descriptor set layouts
        let cmd_list_set_layout = unsafe {
            let bindings = [
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build()
            ];
            
            let create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        
            device
                .create_descriptor_set_layout(&create_info, None)
                .context("Failed to create canvas command list buffer descriptor set layout")?
        };
        
        let image_set_layout = unsafe {
            let bindings = [
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build()
            ];
            
            let create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        
            device
                .create_descriptor_set_layout(&create_info, None)
                .context("Failed to create swapchain image descriptor set layout")?
        };
        
        // Create descriptor pool
        // Number of UNIFORM_BUFFER descriptors = 1 per frame in flight
        // Number of STORAGE_IMAGE descriptors = number of swapchain images
        // Number of descriptor sets = frames in flight + number of swapchain images
        let num_swap_images = frame_queue.swap_image_views().len();
        
        let desc_pool = unsafe {
            let pool_sizes = [
                vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(frames_in_flight)
                    .build(),
    
                vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(num_swap_images as u32)
                    .build()
            ];
            
            let create_info = vk::DescriptorPoolCreateInfo::builder()
                .max_sets(frames_in_flight + num_swap_images as u32)
                .pool_sizes(&pool_sizes);
    
            device
                .create_descriptor_pool(&create_info, None)
                .context("Failed to create descriptor pool")?  
        };
        
        // Allocate descriptor sets
        let cmd_list_desc_sets = unsafe {
            let set_layouts = vec![cmd_list_set_layout; frames_in_flight as usize];

            let alloc_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(desc_pool)
                .set_layouts(&set_layouts);
                
            device
                .allocate_descriptor_sets(&alloc_info)
                .context("Failed to allocate canvas command list descriptor sets")?
        };
        
        let image_desc_sets = unsafe {
            let set_layouts = vec![image_set_layout; num_swap_images];

            let alloc_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(desc_pool)
                .set_layouts(&set_layouts);
                
            device
                .allocate_descriptor_sets(&alloc_info)
                .context("Failed to allocate swapchain image descriptor sets")?
        };
        
        // Update descriptor sets
        // Update command list descriptor sets
        unsafe {
            let buf_infos = cmd_list_bufs
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
                
            let writes = cmd_list_desc_sets
                .iter()
                .zip(&buf_infos)
                .map(|(desc_set, buf_info)| {
                    vk::WriteDescriptorSet::builder()
                        .dst_set(*desc_set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(buf_info)
                        .build()
                })
                .collect::<Vec<_>>();
                
            device.update_descriptor_sets(&writes, &[]);
        }
        
        // Update swapchain image descriptor sets
        unsafe {
            let image_infos = frame_queue
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
                .zip(&image_infos)
                .map(|(desc_set, image_info)| {
                    vk::WriteDescriptorSet::builder()
                        .dst_set(*desc_set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(image_info)
                        .build()
                })
                .collect::<Vec<_>>();
                
            device.update_descriptor_sets(&writes, &[]);
        }
        
        // Create compute pipeline layout
        let pipeline_layout = unsafe {
            let set_layouts = [cmd_list_set_layout, image_set_layout];
            let create_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts);
            
            device
                .create_pipeline_layout(&create_info, None)
                .context("Failed to create pipeline layout")?
        };
        
        // Create shader module
        let shader_module = unsafe {
            let shader_spv = include_bytes!(concat!(
                "..", env!("PATH_SEPERATOR"),
                "..", env!("PATH_SEPERATOR"),
                "..", env!("PATH_SEPERATOR"),
                "shaders", env!("PATH_SEPERATOR"),
                "canvas_2d.spv"
            )).as_slice();
            
            // Convert [u8] to [u32]
            let shader_spv = {
                let len = shader_spv.len() / 4;
                slice::from_raw_parts(shader_spv.as_ptr() as *const u32, len)
            };
            
            let create_info = vk::ShaderModuleCreateInfo::builder().code(shader_spv);
            
            device
                .create_shader_module(&create_info, None)
                .context("Failed to create shader module")?
        };
        
        // Create compute pipeline
        let pipeline = unsafe {
            // Set workgroup size specialization constant
            let spec_consts_buf = [WG_SIZE];
            
            // Convert [u32] to [u8]
            let spec_consts_buf = {
                let len = spec_consts_buf.len() * 4;
                slice::from_raw_parts(spec_consts_buf.as_ptr() as *const u8, len)
            };
            
            let spec_consts_entries = [
                vk::SpecializationMapEntry::builder()
                    .constant_id(0)
                    .offset(0)
                    .size(4)
                    .build()
            ];
            
            let spec_info = vk::SpecializationInfo::builder()
                .map_entries(&spec_consts_entries)
                .data(spec_consts_buf);
    
            let entry_point = CString::new("main").unwrap();
    
            let stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(&entry_point)
                .specialization_info(&spec_info)
                .build();
    
            let create_info = vk::ComputePipelineCreateInfo::builder()
                .stage(stage_create_info)
                .layout(pipeline_layout)
                .build();
                
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None)
                .map_err(|(_, result)| result)
                .context("Failed to create canvas compute pipeline")?[0]
        };
        
        // Destroy unneeded objects
        unsafe {
            device.destroy_shader_module(shader_module, None);
            device.destroy_descriptor_set_layout(cmd_list_set_layout, None);
            device.destroy_descriptor_set_layout(image_set_layout, None);
        }
        
        Ok(Self {
            cmd_list_bufs,
            desc_pool,
            cmd_list_desc_sets,
            image_desc_sets,
            pipeline_layout,
            pipeline
        })
    }
    
    pub fn cmd_render(
        &mut self,
        device: &Device,
        cmd_buf: vk::CommandBuffer,
        frame_info: &FrameInfo,
        record_fn: impl Fn(Canvas2DRecorder<InitState>) -> Canvas2DRecorder<InitState>
    ) {
        // The resources to use for this frame
        let cmd_list_buf = &self.cmd_list_bufs[frame_info.frame_idx()];
        let cmd_list_desc_set = self.cmd_list_desc_sets[frame_info.frame_idx()];
        let image_desc_set = self.image_desc_sets[frame_info.swap_image_idx()];
        
        // Record canvas commands
        record_fn(Canvas2DRecorder::new(cmd_list_buf.ptr())).end();
        
        unsafe {
            // Transfer canvas command list buffer
            cmd_list_buf.cmd_transfer(device, cmd_buf);
            
            // Bind pipeline and descriptor sets
            device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[cmd_list_desc_set, image_desc_set],
                &[]
            );
            
            // Dispatch workgroups
            let workgroups_x = (frame_info.swap_image_extent().width + WG_SIZE - 1) / WG_SIZE;
            let workgroups_y = (frame_info.swap_image_extent().height + WG_SIZE - 1) / WG_SIZE;
            
            device.cmd_dispatch(cmd_buf, workgroups_x, workgroups_y, 1);
        }
    }
    
    pub fn destroy(self, device: &Device, vma_alloc: &VmaAllocator) {
        for buf in self.cmd_list_bufs {
            buf.destroy(vma_alloc);
        }

        unsafe {
            device.destroy_descriptor_pool(self.desc_pool, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_pipeline(self.pipeline, None);
        }
    }
}