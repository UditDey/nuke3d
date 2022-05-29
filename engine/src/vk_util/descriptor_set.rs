use erupt::{vk, DeviceLoader};
use anyhow::{Result, Context};

pub fn create_descriptor_set_layout(
    device: &DeviceLoader,
    desc_types: &[vk::DescriptorType],
    stage_flags: vk::ShaderStageFlags
) -> Result<vk::DescriptorSetLayout> {
    let bindings = desc_types
        .iter()
        .enumerate()
        .map(|(i, &desc_type)| {
            vk::DescriptorSetLayoutBindingBuilder::new()
                .binding(i as u32)
                .descriptor_type(desc_type)
                .descriptor_count(1)
                .stage_flags(stage_flags)
                .immutable_samplers(&[])
        })
        .collect::<Vec<vk::DescriptorSetLayoutBindingBuilder>>();

    let create_info = vk::DescriptorSetLayoutCreateInfoBuilder::new().bindings(&bindings);

    unsafe {
        device.create_descriptor_set_layout(&create_info, None)
            .result()
            .context("Failed to create descriptor set layout")
    }
}