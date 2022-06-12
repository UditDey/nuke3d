use erupt::vk;

pub const SURFACE_FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;
pub const RENDER_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
pub const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

pub const TEXTURE_4_CHANNEL_FORMAT: vk::Format = vk::Format::R8G8B8A8_UINT;
pub const TEXTURE_1_CHANNEL_FORMAT: vk::Format = vk::Format::R8_UINT;

mod instance;
mod debug_messenger;
mod surface;
mod phys_device;
mod device;
mod render_pass;
mod alloc;
mod frame_queue;
mod image;
mod framebuffer;
mod command_buffer;
mod object_name;
mod buffer;
mod shader_module;
mod pipeline;
mod barrier;

pub use instance::create_instance;
pub use debug_messenger::create_debug_messenger;
pub use surface::create_surface;
pub use phys_device::{pick_physical_device, PhysicalDeviceInfo, DEVICE_EXTS};
pub use device::create_device;
pub use render_pass::{MSAALevel, create_render_pass};
pub use alloc::{VkAllocator, MemoryType, MemoryBlock};
pub use frame_queue::{FrameQueue, FrameInfo};
pub use image::{ImageType, Image, create_image_views};
pub use framebuffer::FramebufferSet;
pub use command_buffer::create_command_buffers;
pub use object_name::name_object;
pub(crate) use object_name::name_multiple;
pub use buffer::{BufferType, Buffer, UploadBuffer};
pub use shader_module::create_shader_module;
pub use pipeline::{create_pipeline_layout, create_compute_pipelines};
pub use barrier::{create_memory_barrier, create_image_barrier};