//! 2D vector graphics system for the renderer

use std::mem;
use std::slice;
use std::ffi::CString;

use ash::vk;
use anyhow::{Result, Context};

use super::vk_core::{VkCore, TransferBuffer};

// Compute shader specialization constants
const WORKGROUP_SIZE: u32 = 16;
const NUM_SAMPLES: u32 = 5;

#[repr(C)]
struct Metadata {
    num_lines: u32
}

/// 2D vector graphics canvas
pub struct Canvas2D {
    lines_bufs: Vec<TransferBuffer>,
    desc_pool: vk::DescriptorPool,
    desc_sets: Vec<vk::DescriptorSet>,
    pipeline: vk::Pipeline
}

impl Canvas2D {
    pub fn new(vk_core: &VkCore) -> Result<Self> {
        let queue_len = vk_core.frame_queue().len();

        // --- Create buffers for shape elements ---
        let create_info = vk::BufferCreateInfo::builder()
            .size(2048)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let lines_bufs = (0..queue_len)
            .map(|_| TransferBuffer::new(vk_core, &create_info))
            .collect::<Result<Vec<TransferBuffer>>>()
            .context("Failed to create lines transfer buffers")?;

        // --- Create descriptor set layout ---
        let bindings = [
            // Lines buffer
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),

            // Output image
            vk::DescriptorSetLayoutBinding::builder()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build()
        ];

        let create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);

        let set_layout = unsafe {
            vk_core
                .device()
                .create_descriptor_set_layout(&create_info, None)
                .context("Failed to create descriptor set layout")?
        };

        // --- Create descriptor pool ---
        // In total we have queue_len number of descriptor sets and each set has
        // 1 storage buffer descriptor and 1 storage image descriptor
        let pool_sizes = [
            vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(queue_len as u32)
                .build(),

            vk::DescriptorPoolSize::builder()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(queue_len as u32)
                .build()
        ];

        let create_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(queue_len as u32)
            .pool_sizes(&pool_sizes);

        let desc_pool = unsafe {
            vk_core
                .device()
                .create_descriptor_pool(&create_info, None)
                .context("Failed to create descriptor pool")?
        };

        // --- Allocate descriptor sets ---
        let set_layouts = vec![set_layout; queue_len];

        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(desc_pool)
            .set_layouts(&set_layouts);

        let desc_sets = unsafe {
            vk_core
                .device()
                .allocate_descriptor_sets(&alloc_info)
                .context("Failed to allocate descriptor sets")?
        };

        // --- Update descriptor sets ---
        let lines_buf_infos = lines_bufs
            .iter()
            .map(|buf| {
                let info = vk::DescriptorBufferInfo::builder()
                    .buffer(buf.buf())
                    .offset(0)
                    .range(buf.size())
                    .build();

                [info]
            })
            .collect::<Vec<_>>();

        let image_infos = vk_core
            .frame_queue()
            .swap_image_views()
            .iter()
            .map(|&view| {
                let info = vk::DescriptorImageInfo::builder()
                    .sampler(vk::Sampler::null())
                    .image_view(view)
                    .image_layout(vk::ImageLayout::GENERAL)
                    .build();

                [info]
            })
            .collect::<Vec<_>>();

        let lines_buf_writes = lines_buf_infos
            .iter()
            .zip(&desc_sets)
            .map(|(info, set)| {
                vk::WriteDescriptorSet::builder()
                    .dst_set(*set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
                    .build()
            });

        let image_writes = image_infos
            .iter()
            .zip(&desc_sets)
            .map(|(info, set)| {
                vk::WriteDescriptorSet::builder()
                    .dst_set(*set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(info)
                    .build()
            });

        let writes = lines_buf_writes
            .chain(image_writes)
            .collect::<Vec<_>>();

        unsafe { vk_core.device().update_descriptor_sets(&writes, &[]) };

        // --- Create pipeline layout ---
        let set_layouts = [set_layout];

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

        let pipeline_layout = unsafe {
            vk_core
                .device()
                .create_pipeline_layout(&create_info, None)
                .context("Failed to create pipeline layout")?
        };

        // --- Create shader module ---
        let shader_spv = include_bytes!(concat!(
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

        let shader_module = unsafe {
            vk_core
                .device()
                .create_shader_module(&create_info, None)
                .context("Failed to create canvas2d shader module")?
        };

        // --- Create compute pipeline ---
        let spec_consts_buf = [WORKGROUP_SIZE, NUM_SAMPLES].as_slice();

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

        let create_infos = [
            vk::ComputePipelineCreateInfo::builder()
                .stage(stage_create_info)
                .layout(pipeline_layout)
                .build()
        ];

        let pipeline = unsafe {
            vk_core
                .device()
                .create_compute_pipelines(vk::PipelineCache::null(), &create_infos, None)
                .map_err(|(_, result)| result)
                .context("Failed to create canvas compute pipeline")?[0]
        };

        // --- Destroy unneeded objects ---
        unsafe {
            vk_core.device().destroy_shader_module(shader_module, None);
            vk_core.device().destroy_pipeline_layout(pipeline_layout, None);
            vk_core.device().destroy_descriptor_set_layout(set_layout, None);
        }

        Ok(Self {
            lines_bufs,
            desc_pool,
            desc_sets,
            pipeline
        })
    }

    pub fn destroy(self, vk_core: &VkCore) {
        for buf in self.lines_bufs {
            buf.destroy(vk_core);
        }

        unsafe {
            vk_core.device().destroy_descriptor_pool(self.desc_pool, None);
            vk_core.device().destroy_pipeline(self.pipeline, None);
        }
    }
}