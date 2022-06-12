use std::ffi::CString;

use erupt::{vk, DeviceLoader};
use anyhow::{Result, Context};

pub fn create_pipeline_layout(
    device: &DeviceLoader,
    stage_flags: vk::ShaderStageFlags,
    set_layouts: &[vk::DescriptorSetLayout],
    push_const_size: usize
) -> Result<vk::PipelineLayout> {
    let push_const_ranges = if push_const_size != 0 {
        let range = vk::PushConstantRangeBuilder::new()
            .stage_flags(stage_flags)
            .offset(0)
            .size(push_const_size as u32);

        vec![range]
    }
    else {
        vec![]
    };

    let create_info = vk::PipelineLayoutCreateInfoBuilder::new()
        .set_layouts(set_layouts)
        .push_constant_ranges(&push_const_ranges);

    unsafe {
        device.create_pipeline_layout(&create_info, None)
            .result()
            .context("Failed to create pipeline layout")
    }
}

pub fn create_compute_pipelines<const L: usize>(
    device: &DeviceLoader,
    configs: &[(vk::ShaderModule, vk::PipelineLayout); L]
) -> Result<Box<[vk::Pipeline; L]>> {
    let name = CString::new("main").unwrap();

    let create_infos = configs
        .iter()
        .map(|(shader_mod, pipeline_layout)| {
            let stage = vk::PipelineShaderStageCreateInfoBuilder::new()
                .stage(vk::ShaderStageFlagBits::COMPUTE)
                .module(*shader_mod)
                .name(&name)
                .build_dangling();

            vk::ComputePipelineCreateInfoBuilder::new()
                .stage(stage)
                .layout(*pipeline_layout)
        })
        .collect::<Vec<vk::ComputePipelineCreateInfoBuilder>>();

    unsafe {
        device.create_compute_pipelines(vk::PipelineCache::null(), &create_infos, None)
            .result()
            .context("Failed to create compute pipelines")
            .map(|pipelines| {
                pipelines
                    .into_boxed_slice()
                    .try_into() // Convert our [vk::Pipeline] into [vk::Pipeline; L]
                    .unwrap() // Should never fail, why would we have an incorrect number of pipelines
            })
    }
}