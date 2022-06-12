// This is a reimplementation of piet_gpu::Renderer
// This fork is used as reference: https://github.com/UditDey/piet-gpu

use std::mem;
use std::ptr;
use std::ops::Deref;

use erupt::{vk, DeviceLoader};
use anyhow::{Result, Context};

use piet_gpu::{PietGpuRenderContext, Config};
use piet_gpu_hal::BufWrite;

use crate::vk_util::{
    VkAllocator, BufferType, Buffer, UploadBuffer, PhysicalDeviceInfo, ImageType, Image,
    create_pipeline_layout, create_shader_module, create_compute_pipelines,
    create_memory_barrier, create_image_barrier
};

const TILE_SIZE: u32 = 16;

const PTCL_INITIAL_ALLOC: usize = 1024;

const SCENE_BUF_SIZE: u64 = 1024 * 1024; // 1 MiB
const MEMORY_BUF_SIZE: u64 = 64 * 1024 * 1024; // 64 MiB
const MEMORY_STG_BUF_SIZE: u64 = 2 * 4; // 2 u32s
const CONFIG_BUF_SIZE: u64 = mem::size_of::<Config>() as u64;

pub const NKGUI_IMAGE_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

const N_GRADIENTS: u64 = 256;
const N_SAMPLES: u64 = 512;
const GRADIENT_BUF_SIZE: u64 = N_GRADIENTS * N_SAMPLES * 4;

const TRANSFORM_WG: u32 = 256;
const TRANSFORM_N_ROWS: u32 = 8;
const TRANSFORM_PART_SIZE: u32 = TRANSFORM_WG * TRANSFORM_N_ROWS;
const TRANSFORM_ROOT_BUF_SIZE: u64 = (TRANSFORM_PART_SIZE * 32) as u64;

const REDUCE_WG: u32 = 128;
const REDUCE_N_ROWS: u32 = 2;
const REDUCE_PART_SIZE: u32 = REDUCE_WG * REDUCE_N_ROWS;

const ROOT_WG: u32 = 256;
const ROOT_N_ROWS: u32 = 8;
const ROOT_PART_SIZE: u32 = ROOT_WG * ROOT_N_ROWS;
const PATH_ROOT_BUF_SIZE: u64 = (ROOT_PART_SIZE * 20) as u64;

const SCAN_WG: u32 = 256;
const SCAN_N_ROWS: u32 = 4;
const SCAN_PART_SIZE: u32 = SCAN_WG * SCAN_N_ROWS;

const CLEAR_WG: u32 = 256;

pub const CLIP_PART_SIZE: u32 = 256;

const DRAW_WG: u32 = 256;
const DRAW_N_ROWS: u32 = 8;
const DRAW_PART_SIZE: u32 = DRAW_WG * DRAW_N_ROWS;
const DRAW_ROOT_BUF_SIZE: u64 = (DRAW_PART_SIZE * 16) as u64;

macro_rules! include_shader {
    ($shader_name:literal) => {
        include_bytes!(concat!("../../../../piet-gpu/piet-gpu/shader/gen/", $shader_name, ".spv"))
    };
}

struct DescriptorSetGroup {
    memory_config_set: vk::DescriptorSet,
    memory_config_scene_set: vk::DescriptorSet,
    transform_full_set: vk::DescriptorSet,
    transform_root_set: vk::DescriptorSet,
    path_full_set: vk::DescriptorSet,
    path_root_set: vk::DescriptorSet,
    draw_full_set: vk::DescriptorSet,
    draw_root_set: vk::DescriptorSet,
    fine_raster_set: vk::DescriptorSet
}

pub struct NkGuiRenderer {
    scene_bufs: Vec<UploadBuffer>,
    config_bufs: Vec<UploadBuffer>,
    mem_bufs: Vec<Buffer>,
    mem_stg_bufs: Vec<Buffer>,
    transform_root_bufs: Vec<Buffer>,
    path_root_bufs: Vec<Buffer>,
    draw_root_bufs: Vec<Buffer>,
    gradient_bufs: Vec<Buffer>,
    gradient_images: Vec<Image>,
    bg_image: Image,
    render_images: Vec<Image>,
    render_image_extent: vk::Extent2D,

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

    desc_pool: vk::DescriptorPool,
    desc_set_groups: Vec<DescriptorSetGroup>,

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
        swap_image_extent: vk::Extent2D,
        queue_len: usize,
        cmd_buf: vk::CommandBuffer,
        gfx_queue: vk::Queue,
        vk_alloc: &mut VkAllocator
    ) -> Result<Self> {
        // Create scene buffers, describes what to draw
        let scene_bufs = (0..queue_len)
            .map(|_| UploadBuffer::new(
                device,
                vk_alloc,
                BufferType::ComputeStorage,
                SCENE_BUF_SIZE
            ))
            .collect::<Result<Vec<UploadBuffer>>>()
            .context("Failed to create scene buffers")?;

        // Create config buffer, config info for the shaders to use
        let config_bufs = (0..queue_len)
            .map(|_| UploadBuffer::new(
                device,
                vk_alloc,
                BufferType::ComputeStorage,
                CONFIG_BUF_SIZE
            ))
            .collect::<Result<Vec<UploadBuffer>>>()
            .context("Failed to create config buffers")?;

        // Create memory buffer, storage space for shaders to use
        let mem_bufs = (0..queue_len)
            .map(|_| Buffer::new(
                device,
                vk_alloc,
                BufferType::ComputeStorage,
                MEMORY_BUF_SIZE
            ))
            .collect::<Result<Vec<Buffer>>>()
            .context("Failed to create memory buffers")?;

        let mem_stg_bufs = (0..queue_len)
            .map(|_| Buffer::new(
                device,
                vk_alloc,
                BufferType::Staging,
                MEMORY_STG_BUF_SIZE
            ))
            .collect::<Result<Vec<Buffer>>>()
            .context("Failed to create memory staging buffers")?;

        // Create transform root buffer
        let transform_root_bufs = (0..queue_len)
            .map(|_| Buffer::new(
                device,
                vk_alloc,
                BufferType::ComputeStorage,
                TRANSFORM_ROOT_BUF_SIZE
            ))
            .collect::<Result<Vec<Buffer>>>()
            .context("Failed to create transform root buffers")?;

        // Create path root buffer
        let path_root_bufs = (0..queue_len)
            .map(|_| Buffer::new(
                device,
                vk_alloc,
                BufferType::ComputeStorage,
                PATH_ROOT_BUF_SIZE
            ))
            .collect::<Result<Vec<Buffer>>>()
            .context("Failed to create path root buffers")?;

        // Create draw root buffer
        let draw_root_bufs = (0..queue_len)
            .map(|_| Buffer::new(
                device,
                vk_alloc,
                BufferType::ComputeStorage,
                DRAW_ROOT_BUF_SIZE
            ))
            .collect::<Result<Vec<Buffer>>>()
            .context("Failed to create draw root buffers")?;

        // Create gradient buffer and image, contains gradient data
        let gradient_bufs = (0..queue_len)
            .map(|_| Buffer::new(
                device,
                vk_alloc,
                BufferType::Staging,
                GRADIENT_BUF_SIZE
            ))
            .collect::<Result<Vec<Buffer>>>()
            .context("Failed to create gradient buffers")?;

        let gradient_images = (0..queue_len)
            .map(|_| Image::new(
                device,
                vk_alloc,
                ImageType::NkGuiImage,
                &vk::Extent2D { width: N_SAMPLES as u32, height: N_GRADIENTS as u32 }
            ))
            .collect::<Result<Vec<Image>>>()
            .context("Failed to create gradient images")?;

        // Create background image
        let bg_image = Image::new(
            device,
            vk_alloc,
            ImageType::NkGuiImage,
            &vk::Extent2D { width: 256, height: 256 }
        ).context("Failed to create background image")?;

        // Create render image, rendering will be done on this image
        // Render image size will be the swap image size rounded up to tile alignment
        let render_image_extent = vk::Extent2D {
            width: 2 * (swap_image_extent.width + (swap_image_extent.width.wrapping_neg() & (TILE_SIZE - 1))),
            height: 2 * (swap_image_extent.height + (swap_image_extent.height.wrapping_neg() & (TILE_SIZE - 1))),
        };

        let render_images = (0..queue_len)
            .map(|_| Image::new(
                device,
                vk_alloc,
                ImageType::NkGuiImage,
                &render_image_extent
            ))
            .collect::<Result<Vec<Image>>>()
            .context("Failed to create render images")?;

        // Transition gradient images to TRANSFER_DST_OPTIMAL, and background images and render images
        // to GENERAL
        unsafe {
            let cmd_buf_begin_info = vk::CommandBufferBeginInfoBuilder::new();

            device.begin_command_buffer(cmd_buf, &cmd_buf_begin_info)
                .result()
                .context("Failed to start command buffer recording")?;

            let gradient_image_barriers = gradient_images
                .iter()
                .map(|image| create_image_barrier(
                    image.image(),
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL
                ));

            let bg_image_barriers = [
                create_image_barrier(
                    bg_image.image(),
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::GENERAL
                )
            ].into_iter();

            let render_image_barriers = render_images   
                .iter()
                .map(|image| create_image_barrier(
                    image.image(),
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::GENERAL
                ));

            let barriers = gradient_image_barriers
                .chain(bg_image_barriers)
                .chain(render_image_barriers)
                .collect::<Vec<vk::ImageMemoryBarrierBuilder>>();

            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers
            );

            device.end_command_buffer(cmd_buf)
                .result()
                .context("Failed to end command buffer recording")?;

            let cmd_bufs = [cmd_buf];

            let submit_info = vk::SubmitInfoBuilder::new().command_buffers(&cmd_bufs);

            device.queue_submit(
                gfx_queue,
                &[submit_info],
                vk::Fence::null()
            )
            .result()
            .context("Failed to submit command buffer")?;

            device.queue_wait_idle(gfx_queue)
                .result()
                .context("Failed to wait for queue idle")?;
        }

        // Setup the different stages
        // Descriptor set layouts and pipeline layouts, shared by all the pipelines
        let create_set_layout = |desc_types: &[vk::DescriptorType]| {
            let bindings = desc_types
                .iter()
                .enumerate()
                .map(|(i, &desc_type)| {
                    vk::DescriptorSetLayoutBindingBuilder::new()
                        .binding(i as u32)
                        .descriptor_type(desc_type)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect::<Vec<vk::DescriptorSetLayoutBindingBuilder>>();

            let create_info = vk::DescriptorSetLayoutCreateInfoBuilder::new().bindings(&bindings);

            unsafe { device.create_descriptor_set_layout(&create_info, None) }
                .result()
                .context("Failed to create descriptor set layout")
        };

        // Single bound buffer
        let set_layout_1_buf = create_set_layout(&[vk::DescriptorType::STORAGE_BUFFER])?;

        let pipeline_layout_1_buf = create_pipeline_layout(
            device,
            vk::ShaderStageFlags::COMPUTE,
            &[set_layout_1_buf],
            0
        )?;

        // 2 bound buffers
        let set_layout_2_buf = create_set_layout(&[vk::DescriptorType::STORAGE_BUFFER; 2])?;

        let pipeline_layout_2_buf = create_pipeline_layout(
            device,
            vk::ShaderStageFlags::COMPUTE,
            &[set_layout_2_buf],
            0
        )?;

        // 3 bound buffers
        let set_layout_3_buf = create_set_layout(&[vk::DescriptorType::STORAGE_BUFFER; 3])?;

        let pipeline_layout_3_buf = create_pipeline_layout(
            device,
            vk::ShaderStageFlags::COMPUTE,
            &[set_layout_3_buf],
            0
        )?;

        // 4 bound buffers
        let set_layout_4_buf = create_set_layout(&[vk::DescriptorType::STORAGE_BUFFER; 4])?;

        let pipeline_layout_4_buf = create_pipeline_layout(
            device,
            vk::ShaderStageFlags::COMPUTE,
            &[set_layout_4_buf],
            0
        )?;

        // For the fine raster stage, 2 buffers and 3 images
        let fine_raster_set_layout = create_set_layout(&[
            vk::DescriptorType::STORAGE_BUFFER,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::DescriptorType::STORAGE_IMAGE,
            vk::DescriptorType::STORAGE_IMAGE,
            vk::DescriptorType::STORAGE_IMAGE
        ])?;

        let fine_raster_pipeline_layout = create_pipeline_layout(
            device,
            vk::ShaderStageFlags::COMPUTE,
            &[fine_raster_set_layout],
            0
        )?;

        // Create descriptor sets
        // We make 1 DescriptorSetGroup for each frame in flight
        // Each DescriptorSetGroup has 9 descriptor sets totalling to 22 storage buffer descriptors
        // and 3 storage image descriptors
        // Make descriptor pool accordingly
        let pool_sizes = [
            vk::DescriptorPoolSizeBuilder::new()
                ._type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(22 * queue_len as u32),

            vk::DescriptorPoolSizeBuilder::new()
                ._type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(3 * queue_len as u32)
        ];

        let create_info = vk::DescriptorPoolCreateInfoBuilder::new()
            .max_sets(9 * queue_len as u32)
            .pool_sizes(&pool_sizes);

        let desc_pool = unsafe { device.create_descriptor_pool(&create_info, None) }
            .result()
            .context("Failed to create descriptor pool")?;

        // Allocate DescriptorSetGroups
        let set_layouts = [
            set_layout_2_buf, // For memory_config_set
            set_layout_3_buf, // For memory_config_scene_set
            set_layout_4_buf, // For transform_full_set
            set_layout_1_buf, // For transform_root_set
            set_layout_4_buf, // For path_full_set
            set_layout_1_buf, // For path_root_set
            set_layout_4_buf, // For draw_full_set
            set_layout_1_buf, // For draw_root_set
            fine_raster_set_layout, // For fine_raster_set
        ];

        let desc_set_groups = (0..queue_len)
            .map(|_| {
                let alloc_info = vk::DescriptorSetAllocateInfoBuilder::new()
                    .descriptor_pool(desc_pool)
                    .set_layouts(&set_layouts);

                let desc_sets = unsafe { device.allocate_descriptor_sets(&alloc_info) }
                    .result()
                    .context("Failed to allocate descriptor sets")?;

                let desc_sets: &[vk::DescriptorSet; 9] = desc_sets
                    .as_slice()
                    .try_into() // Convert our &[vk::DescriptorSet] into &[vk::DescriptorSet; 9]
                    .unwrap(); // Should never fail, why would we have an incorrect number of sets

                let &[
                    memory_config_set,
                    memory_config_scene_set,
                    transform_full_set,
                    transform_root_set,
                    path_full_set,
                    path_root_set,
                    draw_full_set,
                    draw_root_set,
                    fine_raster_set
                ] = desc_sets.deref();

                Ok(DescriptorSetGroup {
                    memory_config_set,
                    memory_config_scene_set,
                    transform_full_set,
                    transform_root_set,
                    path_full_set,
                    path_root_set,
                    draw_full_set,
                    draw_root_set,
                    fine_raster_set
                })
            })
            .collect::<Result<Vec<DescriptorSetGroup>>>()?;

        // Fill in our descriptor sets
        for i in 0..queue_len {
            let desc_set_group = &desc_set_groups[i];
            let mem_buf = &mem_bufs[i];
            let config_buf = &config_bufs[i];
            let scene_buf = &scene_bufs[i];
            let transform_root_buf = &transform_root_bufs[i];
            let path_root_buf = &path_root_bufs[i];
            let draw_root_buf = &draw_root_bufs[i];
            let gradient_image = &gradient_images[i];
            let render_image = &render_images[i];

            let buf_infos = [
                // 0 - Memory buffer
                vk::DescriptorBufferInfoBuilder::new()
                    .buffer(mem_buf.buf())
                    .offset(0)
                    .range(MEMORY_BUF_SIZE),

                // 1 - Config buffer
                vk::DescriptorBufferInfoBuilder::new()
                    .buffer(config_buf.target_buf())
                    .offset(0)
                    .range(CONFIG_BUF_SIZE),

                // 2 - Scene buffer
                vk::DescriptorBufferInfoBuilder::new()
                    .buffer(scene_buf.target_buf())
                    .offset(0)
                    .range(SCENE_BUF_SIZE),

                // 3 - Transform root buffer
                vk::DescriptorBufferInfoBuilder::new()
                    .buffer(transform_root_buf.buf())
                    .offset(0)
                    .range(TRANSFORM_ROOT_BUF_SIZE),

                // 4 - Path root buffer
                vk::DescriptorBufferInfoBuilder::new()
                    .buffer(path_root_buf.buf())
                    .offset(0)
                    .range(PATH_ROOT_BUF_SIZE),

                // 5 - Draw root buffer
                vk::DescriptorBufferInfoBuilder::new()
                    .buffer(draw_root_buf.buf())
                    .offset(0)
                    .range(DRAW_ROOT_BUF_SIZE)
            ];

            let image_infos = [
                // 0 - Render image
                vk::DescriptorImageInfoBuilder::new()
                    .sampler(vk::Sampler::null())
                    .image_view(render_image.view())
                    .image_layout(vk::ImageLayout::GENERAL),

                // 1 - Background image
                vk::DescriptorImageInfoBuilder::new()
                    .sampler(vk::Sampler::null())
                    .image_view(bg_image.view())
                    .image_layout(vk::ImageLayout::GENERAL),

                // 0 - Gradient image
                vk::DescriptorImageInfoBuilder::new()
                    .sampler(vk::Sampler::null())
                    .image_view(gradient_image.view())
                    .image_layout(vk::ImageLayout::GENERAL),
            ];

            let desc_writes = [
                // For memory_config_set
                vk::WriteDescriptorSetBuilder::new()
                    .dst_set(desc_set_group.memory_config_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buf_infos[0..2]),

                // For memory_config_scene_set
                vk::WriteDescriptorSetBuilder::new()
                    .dst_set(desc_set_group.memory_config_scene_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buf_infos[0..3]),

                // For transform_full_set
                vk::WriteDescriptorSetBuilder::new()
                    .dst_set(desc_set_group.transform_full_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buf_infos[0..4]),

                // For transform_root_set
                vk::WriteDescriptorSetBuilder::new()
                    .dst_set(desc_set_group.transform_root_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buf_infos[3..4]),

                // For path_full_set
                vk::WriteDescriptorSetBuilder::new() // Bind memory, config and scene buffer
                    .dst_set(desc_set_group.path_full_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buf_infos[0..3]),

                vk::WriteDescriptorSetBuilder::new() // Bind path_root buffer
                    .dst_set(desc_set_group.path_full_set)
                    .dst_binding(3)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buf_infos[4..5]),

                // For path_root_set
                vk::WriteDescriptorSetBuilder::new()
                    .dst_set(desc_set_group.path_root_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buf_infos[4..5]),

                // For draw_full_set
                vk::WriteDescriptorSetBuilder::new() // Bind memory, config and scene buffer
                    .dst_set(desc_set_group.draw_full_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buf_infos[0..3]),

                vk::WriteDescriptorSetBuilder::new() // Bind draw_root buffer
                    .dst_set(desc_set_group.draw_full_set)
                    .dst_binding(3)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buf_infos[5..6]),

                // For draw_root_set
                vk::WriteDescriptorSetBuilder::new()
                    .dst_set(desc_set_group.draw_root_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buf_infos[5..6]),

                // For fine_raster_set
                vk::WriteDescriptorSetBuilder::new() // Bind memory and config buffer
                    .dst_set(desc_set_group.fine_raster_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buf_infos[0..2]),

                vk::WriteDescriptorSetBuilder::new() // Bind render, background and gradient image
                    .dst_set(desc_set_group.fine_raster_set)
                    .dst_binding(2)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&image_infos)
            ];

            unsafe { device.update_descriptor_sets(&desc_writes, &[]); }
        }

        // Create pipelines
        // Transform reduce stage
        let transform_reduce_mod = create_shader_module(device, include_shader!("transform_reduce"))?;

        // Transform root stage
        let transform_root_mod = create_shader_module(device, include_shader!("transform_root"))?;

        // Transform leaf stage
        let transform_leaf_mod = create_shader_module(device, include_shader!("transform_leaf"))?;

        // Pathtag reduce stage
        let pathtag_reduce_mod = create_shader_module(device, include_shader!("pathtag_reduce"))?;

        // Pathtag root stage
        let pathtag_root_mod = create_shader_module(device, include_shader!("pathtag_root"))?;

        // bbox clear stage
        let bbox_clear_mod = create_shader_module(device, include_shader!("bbox_clear"))?;

        // Pathseg stage
        let pathseg_mod = create_shader_module(device, include_shader!("pathseg"))?;

        // Draw reduce stage
        let draw_reduce_mod = create_shader_module(device, include_shader!("draw_reduce"))?;

        // Draw root stage
        let draw_root_mod = create_shader_module(device, include_shader!("draw_root"))?;

        // Draw leaf stage
        let draw_leaf_mod = create_shader_module(device, include_shader!("draw_leaf"))?;

        // Clip reduce stage
        let clip_reduce_mod = create_shader_module(device, include_shader!("clip_reduce"))?;

        // Clip leaf stage
        let clip_leaf_mod = create_shader_module(device, include_shader!("clip_leaf"))?;

        // Tile alloc stage
        let tile_alloc_mod = create_shader_module(device, include_shader!("tile_alloc"))?;

        // Path alloc stage
        let path_alloc_mod = create_shader_module(device, include_shader!("path_coarse"))?;

        // Backdrop propagation stage
        let backdrop_spirv = if phys_dev_info.props().limits.max_compute_work_group_invocations >= 1024 {
            include_shader!("backdrop_lg") as &[u8]
        }
        else {
            include_shader!("backdrop") as &[u8]
        };

        let backdrop_mod = create_shader_module(device, backdrop_spirv)?;

        // Binning stage
        let bin_mod = create_shader_module(device, include_shader!("binning"))?;

        // Coarse raster stage
        let coarse_mod = create_shader_module(device, include_shader!("coarse"))?;

        // Fine raster stage
        let fine_mod = create_shader_module(device, include_shader!("kernel4"))?;

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
            mem_stg_bufs,
            transform_root_bufs,
            path_root_bufs,
            draw_root_bufs,
            gradient_bufs,
            gradient_images,
            bg_image,
            render_images,
            render_image_extent,

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

            desc_pool,
            desc_set_groups,

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

    pub fn cmd_render(
        &mut self,
        device: &DeviceLoader,
        cmd_buf: vk::CommandBuffer,
        render_ctx: &mut PietGpuRenderContext,
        frame_idx: usize
    ) -> Result<()> {
        // Prepare render data
        let (mut config, mut alloc) = render_ctx.stage_config();

        let n_drawobj = render_ctx.n_drawobj();
        let n_path = render_ctx.n_path();
        let n_transform = render_ctx.n_transform() as u32;
        let n_drawobj = render_ctx.n_drawobj() as u32;
        let n_pathseg = render_ctx.n_pathseg();
        let n_pathtag = render_ctx.n_pathtag() as u32;
        let n_clip = render_ctx.n_clip();
        let ramp_data = render_ctx.get_ramp_data();

        const PATH_SIZE: u32 = 12;
        const BIN_SIZE: u32 = 8;
        
        let width_in_tiles = self.render_image_extent.width as usize / TILE_SIZE as usize;
        let height_in_tiles = self.render_image_extent.height as usize / TILE_SIZE as usize;
        let tile_base = alloc;

        alloc += (((n_path + 3) & !3) * PATH_SIZE) as usize;

        let bin_base = alloc;
        alloc += (((n_drawobj + 255) & !255) * BIN_SIZE) as usize;

        let ptcl_base = alloc;
        alloc += width_in_tiles * height_in_tiles * PTCL_INITIAL_ALLOC;

        config.width_in_tiles = width_in_tiles as u32;
        config.height_in_tiles = height_in_tiles as u32;
        config.tile_alloc = tile_base as u32;
        config.bin_alloc = bin_base as u32;
        config.ptcl_alloc = ptcl_base as u32;

        let scene_buf = &self.scene_bufs[frame_idx];
        let config_buf = &self.config_bufs[frame_idx];
        let mem_stg_buf = &self.mem_stg_bufs[frame_idx];
        let mem_buf = &self.mem_bufs[frame_idx];
        let gradient_buf = &self.gradient_bufs[frame_idx];
        let gradient_image = &self.gradient_images[frame_idx];
        let desc_set_group = &self.desc_set_groups[frame_idx];

        // Copy scene data
        let mut scene_buf_write = BufWrite::new(scene_buf.ptr() as *mut u8, 0, SCENE_BUF_SIZE as usize);
        render_ctx.write_scene(&mut scene_buf_write);

        unsafe {
            // Copy config data
            (config_buf.ptr() as *mut Config).write(config);

            // Copy memory buffer data
            let mem_stg_buf_ptr = mem_stg_buf.ptr()? as *mut u32;

            mem_stg_buf_ptr.write(alloc as u32);
            mem_stg_buf_ptr.offset(1).write(0);

            // Copy gradient data
            ptr::copy_nonoverlapping(ramp_data.as_ptr(), gradient_buf.ptr()? as *mut u32, ramp_data.len());
        }

        // Record render commands
        unsafe {
            // Upload render data
            scene_buf.cmd_upload(device, cmd_buf);
            config_buf.cmd_upload(device, cmd_buf);

            let copy_region = vk::BufferCopyBuilder::new()
                .src_offset(0)
                .dst_offset(0)
                .size(MEMORY_STG_BUF_SIZE);

            device.cmd_copy_buffer(cmd_buf, mem_stg_buf.buf(), mem_buf.buf(), &[copy_region]);

            let copy_region = vk::BufferImageCopyBuilder::new()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: N_SAMPLES as u32,
                    height: N_GRADIENTS as u32,
                    depth: 1
                });

            device.cmd_copy_buffer_to_image(
                cmd_buf,
                gradient_buf.buf(),
                gradient_image.image(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy_region]
            );

            // Wait for uploads to finish and transition gradient image to GENERAL
            let mem_barrier = create_memory_barrier();

            let img_barrier = create_image_barrier(
                gradient_image.image(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::GENERAL
            );

            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[mem_barrier],
                &[],
                &[img_barrier]
            );

            // Start rendering
            let cmd_bind_pipeline = |pipeline| {
                device.cmd_bind_pipeline(
                    cmd_buf,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline
                );
            };

            let cmd_bind_desc_set = |desc_set, desc_layout| {
                device.cmd_bind_descriptor_sets(
                    cmd_buf,
                    vk::PipelineBindPoint::COMPUTE,
                    desc_layout,
                    0,
                    &[desc_set],
                    &[]
                );
            };

            let cmd_dispatch = |workgroups_x, workgroups_y| {
                device.cmd_dispatch(cmd_buf, workgroups_x, workgroups_y, 1);
            };

            let cmd_memory_barrier = || {
                let barrier = create_memory_barrier();
                
                device.cmd_pipeline_barrier(
                    cmd_buf,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[]
                );
            };


            // Transform stage
            let n_workgroups = (n_transform + TRANSFORM_PART_SIZE - 1) / TRANSFORM_PART_SIZE;

            if n_workgroups > 1 {
                // Transform reduce
                cmd_bind_pipeline(self.transform_reduce_pipeline);
                cmd_bind_desc_set(desc_set_group.transform_full_set, self.pipeline_layout_4_buf);                
                cmd_dispatch(n_workgroups, 1);

                cmd_memory_barrier();

                // Transform root
                cmd_bind_pipeline(self.transform_root_pipeline);
                cmd_bind_desc_set(desc_set_group.transform_root_set, self.pipeline_layout_1_buf);                
                cmd_dispatch(1, 1);

                cmd_memory_barrier();
            }

            // Transform leaf
            cmd_bind_pipeline(self.tranform_leaf_pipeline);
            cmd_bind_desc_set(desc_set_group.transform_full_set, self.pipeline_layout_4_buf);                
            cmd_dispatch(n_workgroups, 1);


            // Path stage
            let reduce_part_tags = REDUCE_PART_SIZE * 4;
            let n_wg_tag_reduce = (n_pathtag + reduce_part_tags * 4 - 1) / reduce_part_tags * 4;

            if n_wg_tag_reduce > 1 {
                // Path reduce
                cmd_bind_pipeline(self.pathtag_reduce_pipeline);
                cmd_bind_desc_set(desc_set_group.path_full_set, self.pipeline_layout_4_buf);
                cmd_dispatch(n_wg_tag_reduce, 1);

                cmd_memory_barrier();

                // Path root
                cmd_bind_pipeline(self.pathtag_root_pipeline);
                cmd_bind_desc_set(desc_set_group.path_root_set, self.pipeline_layout_1_buf);
                cmd_dispatch(1, 1);
            }

            // bbox clear
            let n_wg_clear = (n_path + CLEAR_WG - 1) / CLEAR_WG;

            cmd_bind_pipeline(self.bbox_clear_pipeline);
            cmd_bind_desc_set(desc_set_group.memory_config_set, self.pipeline_layout_2_buf);
            cmd_dispatch(n_wg_clear, 1);

            cmd_memory_barrier();

            // Pathseg
            let n_wg_pathseg = (n_pathtag + SCAN_PART_SIZE - 1) / SCAN_PART_SIZE;

            cmd_bind_pipeline(self.pathseg_pipeline);
            cmd_bind_desc_set(desc_set_group.path_full_set, self.pipeline_layout_4_buf);
            cmd_dispatch(n_wg_pathseg, 1);

            // Draw stage
            let n_workgroups = (n_drawobj + DRAW_PART_SIZE - 1) / DRAW_PART_SIZE;

            if n_workgroups > 1 {
                // Draw reduce
                cmd_bind_pipeline(self.draw_reduce_pipeline);
                cmd_bind_desc_set(desc_set_group.draw_full_set, self.pipeline_layout_4_buf);
                cmd_dispatch(n_workgroups, 1);

                cmd_memory_barrier();

                // Draw root
                cmd_bind_pipeline(self.draw_root_pipeline);
                cmd_bind_desc_set(desc_set_group.draw_root_set, self.pipeline_layout_1_buf);
                cmd_dispatch(1, 1);
            }

            cmd_memory_barrier();

            // Draw leaf
            cmd_bind_pipeline(self.draw_leaf_pipeline);
            cmd_bind_desc_set(desc_set_group.draw_full_set, self.pipeline_layout_4_buf);
            cmd_dispatch(n_workgroups, 1);

            cmd_memory_barrier();

            // Clip reduce
            let n_wg_reduce = n_clip.saturating_sub(1) / CLIP_PART_SIZE;

            if n_wg_reduce > 0 {
                cmd_bind_pipeline(self.clip_reduce_pipeline);
                cmd_bind_desc_set(desc_set_group.memory_config_set, self.pipeline_layout_2_buf);
                cmd_dispatch(n_wg_reduce, 1);

                cmd_memory_barrier();
            }

            // Clip leaf
            let n_wg = (n_clip + CLIP_PART_SIZE - 1) / CLIP_PART_SIZE;

            if n_wg > 0 {
                cmd_bind_pipeline(self.clip_leaf_pipeline);
                cmd_bind_desc_set(desc_set_group.memory_config_set, self.pipeline_layout_2_buf);
                cmd_dispatch(n_wg, 1);

                cmd_memory_barrier();
            }

            // Binning
            let n_workgroups = (n_path + 255) / 256;

            cmd_bind_pipeline(self.bin_pipeline);
            cmd_bind_desc_set(desc_set_group.memory_config_set, self.pipeline_layout_2_buf);
            cmd_dispatch(n_workgroups, 1);

            cmd_memory_barrier();

            // Tile alloc
            cmd_bind_pipeline(self.tile_alloc_pipeline);
            cmd_bind_desc_set(desc_set_group.memory_config_scene_set, self.pipeline_layout_3_buf);
            cmd_dispatch(n_workgroups, 1);

            cmd_memory_barrier();

            // Path flattening
            let n_workgroups = (n_pathseg + 31) / 32;

            cmd_bind_pipeline(self.path_alloc_pipeline);
            cmd_bind_desc_set(desc_set_group.memory_config_set, self.pipeline_layout_2_buf);
            cmd_dispatch(n_workgroups, 1);

            cmd_memory_barrier();

            // Backdrop propagation
            let n_workgroups = (n_path + 255) / 256;

            cmd_bind_pipeline(self.backdrop_pipeline);
            cmd_bind_desc_set(desc_set_group.memory_config_set, self.pipeline_layout_2_buf);
            cmd_dispatch(n_workgroups, 1);

            cmd_memory_barrier();

            // Coarse raster
            let n_workgroups_x = (self.render_image_extent.width + 255) / 256;
            let n_workgroups_y = (self.render_image_extent.height + 255) / 256;

            cmd_bind_pipeline(self.coarse_pipeline);
            cmd_bind_desc_set(desc_set_group.memory_config_scene_set, self.pipeline_layout_3_buf);
            cmd_dispatch(n_workgroups_x, n_workgroups_y);

            cmd_memory_barrier();

            // Fine raster
            let n_workgroups_x = self.render_image_extent.width / TILE_SIZE;
            let n_workgroups_y = self.render_image_extent.height / TILE_SIZE;

            cmd_bind_pipeline(self.fine_pipeline);
            cmd_bind_desc_set(desc_set_group.fine_raster_set, self.fine_raster_pipeline_layout);
            cmd_dispatch(n_workgroups_x, n_workgroups_y);

            cmd_memory_barrier();

            // Transition gradient image back to TRANSFER_DST_OPTIMAL for next use
            let img_barrier = create_image_barrier(
                gradient_image.image(),
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL
            );

            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[img_barrier]
            );
        }
        
        Ok(())
    }

    pub fn render_image(&self, frame_idx: usize) -> vk::Image {
        self.render_images[frame_idx].image()
    }

    pub fn render_image_extent(&self) -> vk::Extent2D {
        self.render_image_extent
    }

    pub fn destroy(self, device: &DeviceLoader, vk_alloc: &mut VkAllocator) {
        let upload_bufs = self.scene_bufs
            .into_iter()
            .chain(self.config_bufs);
        
        let bufs = self.mem_stg_bufs
            .into_iter()
            .chain(self.mem_bufs)
            .chain(self.transform_root_bufs)
            .chain(self.path_root_bufs)
            .chain(self.draw_root_bufs)
            .chain(self.gradient_bufs);

        for buf in upload_bufs {
            buf.destroy(device, vk_alloc);
        }

        for buf in bufs {
            buf.destroy(device, vk_alloc);
        }

        let images = self.gradient_images
            .into_iter()
            .chain(self.render_images);

        for image in images {
            image.destroy(device, vk_alloc);
        }

        self.bg_image.destroy(device, vk_alloc);

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

            device.destroy_descriptor_pool(self.desc_pool, None);

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