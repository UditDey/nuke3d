use erupt::{vk, DeviceLoader};
use anyhow::{Result, Context};

use super::{name_object, RENDER_FORMAT, DEPTH_FORMAT};

#[derive(Clone, Copy, PartialEq)]
pub enum MSAALevel {
    Off,
    Two,
    Four,
    Eight
}

impl MSAALevel {
    pub fn samples(&self) -> vk::SampleCountFlagBits {
        match self {
            Self::Off => vk::SampleCountFlagBits::_1,
            Self::Two => vk::SampleCountFlagBits::_2,
            Self::Four => vk::SampleCountFlagBits::_4,
            Self::Eight => vk::SampleCountFlagBits::_8
        }
    }
}

pub fn create_render_pass(device: &DeviceLoader, msaa_level: MSAALevel) -> Result<vk::RenderPass> {
    let samples = msaa_level.samples();

    // Render pass attachments
    let mut attachments = vec![
        // Render attachment
        vk::AttachmentDescriptionBuilder::new()
            .format(RENDER_FORMAT)
            .samples(samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),

        // Depth attachment
        vk::AttachmentDescriptionBuilder::new()
            .format(DEPTH_FORMAT)
            .samples(samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)  
    ];

    // Attachment references
    let render_ref = vk::AttachmentReferenceBuilder::new()
        .attachment(0)
        .layout(vk::ImageLayout::PRESENT_SRC_KHR);

    let depth_ref = vk::AttachmentReferenceBuilder::new()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    // If MSAA is enabled, we need a resolve attachment as well
    let resolve_ref = if samples != vk::SampleCountFlagBits::_1 {
        let resolve_attachment = vk::AttachmentDescriptionBuilder::new()
            .format(RENDER_FORMAT)
            .samples(vk::SampleCountFlagBits::_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        attachments.push(resolve_attachment);

        let resolve_ref = vk::AttachmentReferenceBuilder::new()
            .attachment(2)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        Some(resolve_ref)
    }
    else {
        None
    };

    let resolve_attachments = resolve_ref
        .as_ref()
        .map(std::slice::from_ref)
        .unwrap_or_default();

    let color_attachments = [render_ref];

    // Subpasses
    let subpasses = [
        // Depth prepass
        vk::SubpassDescriptionBuilder::new()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .depth_stencil_attachment(&depth_ref),

        // Main graphics pass
        vk::SubpassDescriptionBuilder::new()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachments)
            .resolve_attachments(resolve_attachments)
            .depth_stencil_attachment(&depth_ref)
    ];

    let dependencies = [
        // Dependency between start of render pass and depth prepass
        vk::SubpassDependencyBuilder::new()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
            .dst_stage_mask(
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS |
                vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .dependency_flags(vk::DependencyFlags::BY_REGION),

        // Dependency between depth prepass and main graphics pass
        vk::SubpassDependencyBuilder::new()
            .src_subpass(0)
            .dst_subpass(1)
            .src_stage_mask(vk::PipelineStageFlags::BOTTOM_OF_PIPE)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::MEMORY_READ)
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ |
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
            )
            .dependency_flags(vk::DependencyFlags::BY_REGION)
    ];

    let create_info = vk::RenderPassCreateInfoBuilder::new()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);

    let render_pass = unsafe { device.create_render_pass(&create_info, None) }
        .result()
        .context("Failed to create render pass")?;

    name_object(device, render_pass.object_handle(), vk::ObjectType::RENDER_PASS, "render_pass")?;

    Ok(render_pass)
}