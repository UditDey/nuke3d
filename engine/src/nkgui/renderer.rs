// This is a reimplementation of piet-gpu
// This fork is used as reference: https://github.com/UditDey/piet-gpu

use std::{mem, ops::Deref};

use erupt::{vk, DeviceLoader};
use anyhow::{Result, Context};

use crate::vk_util::{
    VkAllocator, BufferType, Buffer, PhysicalDeviceInfo,
    create_descriptor_set_layout, create_pipeline_layout,
    create_shader_module, create_compute_pipelines,
    name_object, name_multiple
};

const TILE_WIDTH: usize = 16;
const TILE_HEIGHT: usize = 16;

const SCENE_BUF_SIZE: u64 = 8 * 1024 * 1024;
const MEMORY_BUF_SIZE: u64 = 128 * 1024 * 1024;

const N_GRADIENTS: u64 = 256;
const GRADIENT_N_SAMPLES: u64 = 512;
const GRADIENT_BUF_SIZE: u64 = N_GRADIENTS * GRADIENT_N_SAMPLES;

#[repr(C)]
pub struct Config {
    pub n_elements: u32,
    pub n_pathseg: u32,
    pub width_in_tiles: u32,
    pub height_in_tiles: u32,
    pub tile_alloc: u32,
    pub bin_alloc: u32,
    pub ptcl_alloc: u32,
    pub pathseg_alloc: u32,
    pub anno_alloc: u32,
    pub trans_alloc: u32,
    pub path_bbox_alloc: u32,
    pub drawmonoid_alloc: u32,
    pub clip_alloc: u32,
    pub clip_bic_alloc: u32,
    pub clip_stack_alloc: u32,
    pub clip_bbox_alloc: u32,
    pub draw_bbox_alloc: u32,
    pub drawinfo_alloc: u32,
    pub n_trans: u32,
    pub n_path: u32,
    pub n_clip: u32,
    pub trans_offset: u32,
    pub linewidth_offset: u32,
    pub pathtag_offset: u32,
    pub pathseg_offset: u32,
    pub drawtag_offset: u32,
    pub drawdata_offset: u32,
}

pub struct NkGuiRenderer {
    scene_bufs: Vec<Buffer>,
    config_bufs: Vec<Buffer>,
    mem_bufs: Vec<Buffer>,
    gradient_bufs: Vec<Buffer>,

    set_layout_1_buf: vk::DescriptorSetLayout,
    set_layout_2_buf: vk::DescriptorSetLayout,
    set_layout_3_buf: vk::DescriptorSetLayout,
    set_layout_4_buf: vk::DescriptorSetLayout,
    fine_raster_set_layout: vk::DescriptorSetLayout,

    pipeline_layout_1_buf: vk::PipelineLayout,
    pipeline_layout_2_buf: vk::PipelineLayout,
    pipeline_layout_3_buf: vk::PipelineLayout,
    pipeline_layout_4_buf: vk::PipelineLayout,
    fine_raster_pipeline_layout: vk::PipelineLayout,

    transform_reduce_pipeline: vk::Pipeline,
    transform_root_pipeline: vk::Pipeline,
    tranform_leaf_pipeline: vk::Pipeline,
    pathtag_reduce_pipeline: vk::Pipeline,
    pathtag_root_pipeline: vk::Pipeline,
    bbox_clear_pipeline: vk::Pipeline,
    pathseg_pipeline: vk::Pipeline,
    draw_reduce_pipeline: vk::Pipeline,
    draw_root_pipeline: vk::Pipeline,
    draw_leaf_pipeline: vk::Pipeline,
    clip_reduce_pipeline: vk::Pipeline,
    clip_leaf_pipeline: vk::Pipeline,
    tile_alloc_pipeline: vk::Pipeline,
    path_alloc_pipeline: vk::Pipeline,
    backdrop_pipeline: vk::Pipeline,
    bin_pipeline: vk::Pipeline,
    coarse_pipeline: vk::Pipeline,
    fine_pipeline: vk::Pipeline
}

impl NkGuiRenderer {
    pub fn new(
        device: &DeviceLoader,
        phys_dev_info: &PhysicalDeviceInfo,
        vk_alloc: &mut VkAllocator,
        queue_len: usize
    ) -> Result<Self> {
        // Create scene buffers, describes what to draw
        let scene_bufs = (0..queue_len)
            .map(|_| Buffer::new(
                device,
                vk_alloc,
                BufferType::ComputeUpload,
                SCENE_BUF_SIZE
            ))
            .collect::<Result<Vec<Buffer>>>()
            .context("Failed to create scene buffers")?;

        // Create config buffer, config info for the shaders to use
        let config_bufs = (0..queue_len)
            .map(|_| Buffer::new(
                device,
                vk_alloc,
                BufferType::ComputeUpload,
                mem::size_of::<Config>() as u64
            ))
            .collect::<Result<Vec<Buffer>>>()
            .context("Failed to create config buffers")?;

        // Create memory buffer, storage buffer for shaders to use
        let mem_bufs = (0..queue_len)
            .map(|_| Buffer::new(
                device,
                vk_alloc,
                BufferType::ComputeUpload,
                MEMORY_BUF_SIZE
            ))
            .collect::<Result<Vec<Buffer>>>()
            .context("Failed to create memory buffers")?;

        // Create gradient buffer, contains gradient data
        let gradient_bufs = (0..queue_len)
            .map(|_| Buffer::new(
                device,
                vk_alloc,
                BufferType::ComputeUpload,
                GRADIENT_BUF_SIZE
            ))
            .collect::<Result<Vec<Buffer>>>()
            .context("Failed to create gradient buffers")?;

        // Setup the different stages

        // Descriptor set layouts and pipeline layouts, shared by all the pipelines
        // Single bound buffer
        let set_layout_1_buf = create_descriptor_set_layout(
            device,
            &[vk::DescriptorType::STORAGE_BUFFER],
            vk::ShaderStageFlags::COMPUTE
        )?;

        let pipeline_layout_1_buf = create_pipeline_layout(
            device,
            vk::ShaderStageFlags::COMPUTE,
            &[set_layout_1_buf],
            0
        )?;

        // 2 bound buffers
        let set_layout_2_buf = create_descriptor_set_layout(
            device,
            &[vk::DescriptorType::STORAGE_BUFFER; 2],
            vk::ShaderStageFlags::COMPUTE
        )?;

        let pipeline_layout_2_buf = create_pipeline_layout(
            device,
            vk::ShaderStageFlags::COMPUTE,
            &[set_layout_2_buf],
            0
        )?;

        // 3 bound buffers
        let set_layout_3_buf = create_descriptor_set_layout(
            device,
            &[vk::DescriptorType::STORAGE_BUFFER; 3],
            vk::ShaderStageFlags::COMPUTE
        )?;

        let pipeline_layout_3_buf = create_pipeline_layout(
            device,
            vk::ShaderStageFlags::COMPUTE,
            &[set_layout_3_buf],
            0
        )?;

        // 4 bound buffers
        let set_layout_4_buf = create_descriptor_set_layout(
            device,
            &[vk::DescriptorType::STORAGE_BUFFER; 4],
            vk::ShaderStageFlags::COMPUTE
        )?;

        let pipeline_layout_4_buf = create_pipeline_layout(
            device,
            vk::ShaderStageFlags::COMPUTE,
            &[set_layout_4_buf],
            0
        )?;

        // For the fine raster stage, 2 buffers and 3 images
        let fine_raster_set_layout = create_descriptor_set_layout(
            device,
            &[
                vk::DescriptorType::STORAGE_BUFFER,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::DescriptorType::STORAGE_IMAGE,
                vk::DescriptorType::STORAGE_IMAGE,
                vk::DescriptorType::STORAGE_IMAGE
            ],
            vk::ShaderStageFlags::COMPUTE
        )?;

        let fine_raster_pipeline_layout = create_pipeline_layout(
            device,
            vk::ShaderStageFlags::COMPUTE,
            &[fine_raster_set_layout],
            0
        )?;

        // Create pipelines
        // Transform reduce stage
        let transform_reduce_mod = create_shader_module(
            device,
            include_bytes!("../../nkgui_shaders/transform_reduce.spv")
        )?;

        // Transform root stage
        let transform_root_mod = create_shader_module(
            device,
            include_bytes!("../../nkgui_shaders/transform_root.spv")
        )?;

        // Transform leaf stage
        let transform_leaf_mod = create_shader_module(
            device,
            include_bytes!("../../nkgui_shaders/transform_leaf.spv")
        )?;

        // Pathtag reduce stage
        let pathtag_reduce_mod = create_shader_module(
            device,
            include_bytes!("../../nkgui_shaders/pathtag_reduce.spv")
        )?;

        // Pathtag root stage
        let pathtag_root_mod = create_shader_module(
            device,
            include_bytes!("../../nkgui_shaders/pathtag_root.spv")
        )?;

        // bbox clear stage
        let bbox_clear_mod = create_shader_module(
            device,
            include_bytes!("../../nkgui_shaders/bbox_clear.spv")
        )?;

        // Pathseg stage
        let pathseg_mod = create_shader_module(
            device,
            include_bytes!("../../nkgui_shaders/pathseg.spv")
        )?;

        // Draw reduce stage
        let draw_reduce_mod = create_shader_module(
            device,
            include_bytes!("../../nkgui_shaders/draw_reduce.spv")
        )?;

        // Draw root stage
        let draw_root_mod = create_shader_module(
            device,
            include_bytes!("../../nkgui_shaders/draw_root.spv")
        )?;

        // Draw leaf stage
        let draw_leaf_mod = create_shader_module(
            device,
            include_bytes!("../../nkgui_shaders/draw_leaf.spv")
        )?;

        // Clip reduce stage
        let clip_reduce_mod = create_shader_module(
            device,
            include_bytes!("../../nkgui_shaders/clip_reduce.spv")
        )?;

        // Clip leaf stage
        let clip_leaf_mod = create_shader_module(
            device,
            include_bytes!("../../nkgui_shaders/clip_leaf.spv")
        )?;

        // Tile alloc stage
        let tile_alloc_mod = create_shader_module(
            device,
            include_bytes!("../../nkgui_shaders/tile_alloc.spv")
        )?;

        // Path alloc stage
        let path_alloc_mod = create_shader_module(
            device,
            include_bytes!("../../nkgui_shaders/path_coarse.spv")
        )?;

        // Backdrop propagation stage
        let backdrop_spirv = if phys_dev_info.props.limits.max_compute_work_group_invocations >= 1024 {
            include_bytes!("../../nkgui_shaders/backdrop_lg.spv") as &[u8]
        }
        else {
            include_bytes!("../../nkgui_shaders/backdrop.spv") as &[u8]
        };

        let backdrop_mod = create_shader_module(device, backdrop_spirv)?;

        // Binning stage
        let bin_mod = create_shader_module(
            device,
            include_bytes!("../../nkgui_shaders/binning.spv")
        )?;

        // Coarse raster stage
        let coarse_mod = create_shader_module(
            device,
            include_bytes!("../../nkgui_shaders/coarse.spv")
        )?;

        // Fine raster stage
        let fine_mod = create_shader_module(
            device,
            include_bytes!("../../nkgui_shaders/kernel4.spv")
        )?;

        let configs = [
            (transform_reduce_mod, pipeline_layout_4_buf),
            (transform_root_mod, pipeline_layout_1_buf),
            (transform_leaf_mod, pipeline_layout_4_buf),
            (pathtag_reduce_mod, pipeline_layout_4_buf),
            (pathtag_root_mod, pipeline_layout_1_buf),
            (bbox_clear_mod, pipeline_layout_2_buf),
            (pathseg_mod, pipeline_layout_4_buf),
            (draw_reduce_mod, pipeline_layout_4_buf),
            (draw_root_mod, pipeline_layout_1_buf),
            (draw_leaf_mod, pipeline_layout_4_buf),
            (clip_reduce_mod, pipeline_layout_2_buf),
            (clip_leaf_mod, pipeline_layout_2_buf),
            (tile_alloc_mod, pipeline_layout_3_buf),
            (path_alloc_mod, pipeline_layout_2_buf),
            (backdrop_mod, pipeline_layout_2_buf),
            (bin_mod, pipeline_layout_2_buf),
            (coarse_mod, pipeline_layout_3_buf),
            (fine_mod, fine_raster_pipeline_layout)
        ];

        let &[
            transform_reduce_pipeline,
            transform_root_pipeline,
            tranform_leaf_pipeline,
            pathtag_reduce_pipeline,
            pathtag_root_pipeline,
            bbox_clear_pipeline,
            pathseg_pipeline,
            draw_reduce_pipeline,
            draw_root_pipeline,
            draw_leaf_pipeline,
            clip_reduce_pipeline,
            clip_leaf_pipeline,
            tile_alloc_pipeline,
            path_alloc_pipeline,
            backdrop_pipeline,
            bin_pipeline,
            coarse_pipeline,
            fine_pipeline
        ] = create_compute_pipelines(device, &configs)?.deref();

        // We're done with the shader modules, destroy them
        unsafe {
            device.destroy_shader_module(transform_reduce_mod, None);
            device.destroy_shader_module(transform_root_mod, None);
            device.destroy_shader_module(transform_leaf_mod, None);
            device.destroy_shader_module(pathtag_reduce_mod, None);
            device.destroy_shader_module(pathtag_root_mod, None);
            device.destroy_shader_module(bbox_clear_mod, None);
            device.destroy_shader_module(pathseg_mod, None);
            device.destroy_shader_module(draw_reduce_mod, None);
            device.destroy_shader_module(draw_root_mod, None);
            device.destroy_shader_module(draw_leaf_mod, None);
            device.destroy_shader_module(clip_reduce_mod, None);
            device.destroy_shader_module(clip_leaf_mod, None);
            device.destroy_shader_module(tile_alloc_mod, None);
            device.destroy_shader_module(path_alloc_mod, None);
            device.destroy_shader_module(backdrop_mod, None);
            device.destroy_shader_module(bin_mod, None);
            device.destroy_shader_module(coarse_mod, None);
            device.destroy_shader_module(fine_mod, None);
        }

        Ok(Self {
            scene_bufs,
            config_bufs,
            mem_bufs,
            gradient_bufs,

            set_layout_1_buf,
            set_layout_2_buf,
            set_layout_3_buf,
            set_layout_4_buf,
            fine_raster_set_layout,

            pipeline_layout_1_buf,
            pipeline_layout_2_buf,
            pipeline_layout_3_buf,
            pipeline_layout_4_buf,
            fine_raster_pipeline_layout,

            transform_reduce_pipeline,
            transform_root_pipeline,
            tranform_leaf_pipeline,
            pathtag_reduce_pipeline,
            pathtag_root_pipeline,
            bbox_clear_pipeline,
            pathseg_pipeline,
            draw_reduce_pipeline,
            draw_root_pipeline,
            draw_leaf_pipeline,
            clip_reduce_pipeline,
            clip_leaf_pipeline,
            tile_alloc_pipeline,
            path_alloc_pipeline,
            backdrop_pipeline,
            bin_pipeline,
            coarse_pipeline,
            fine_pipeline
        })
    }

    pub fn destroy(self, device: &DeviceLoader, vk_alloc: &mut VkAllocator) {
        for buf in self.scene_bufs {
            buf.destroy(device, vk_alloc);
        }

        for buf in self.config_bufs {
            buf.destroy(device, vk_alloc);
        }

        for buf in self.mem_bufs {
            buf.destroy(device, vk_alloc);
        }

        for buf in self.gradient_bufs {
            buf.destroy(device, vk_alloc);
        }

        unsafe {
            device.destroy_descriptor_set_layout(self.set_layout_1_buf, None);
            device.destroy_descriptor_set_layout(self.set_layout_2_buf, None);
            device.destroy_descriptor_set_layout(self.set_layout_3_buf, None);
            device.destroy_descriptor_set_layout(self.set_layout_4_buf, None);
            device.destroy_descriptor_set_layout(self.fine_raster_set_layout, None);

            device.destroy_pipeline_layout(self.pipeline_layout_1_buf, None);
            device.destroy_pipeline_layout(self.pipeline_layout_2_buf, None);
            device.destroy_pipeline_layout(self.pipeline_layout_3_buf, None);
            device.destroy_pipeline_layout(self.pipeline_layout_4_buf, None);
            device.destroy_pipeline_layout(self.fine_raster_pipeline_layout, None);

            device.destroy_pipeline(self.transform_reduce_pipeline, None);
            device.destroy_pipeline(self.transform_root_pipeline, None);
            device.destroy_pipeline(self.tranform_leaf_pipeline, None);
            device.destroy_pipeline(self.pathtag_reduce_pipeline, None);
            device.destroy_pipeline(self.pathtag_root_pipeline, None);
            device.destroy_pipeline(self.bbox_clear_pipeline, None);
            device.destroy_pipeline(self.pathseg_pipeline, None);
            device.destroy_pipeline(self.draw_reduce_pipeline, None);
            device.destroy_pipeline(self.draw_root_pipeline, None);
            device.destroy_pipeline(self.draw_leaf_pipeline, None);
            device.destroy_pipeline(self.clip_reduce_pipeline, None);
            device.destroy_pipeline(self.clip_leaf_pipeline, None);
            device.destroy_pipeline(self.tile_alloc_pipeline, None);
            device.destroy_pipeline(self.path_alloc_pipeline, None);
            device.destroy_pipeline(self.backdrop_pipeline, None);
            device.destroy_pipeline(self.bin_pipeline, None);
            device.destroy_pipeline(self.coarse_pipeline, None);
            device.destroy_pipeline(self.fine_pipeline, None);
        }
    }
}