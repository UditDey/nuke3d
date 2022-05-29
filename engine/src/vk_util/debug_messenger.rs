use std::ffi::CStr;
use std::os::raw::c_void;

use erupt::{vk, InstanceLoader};
use anyhow::{Result, Context};

#[allow(dead_code)]
unsafe extern "system" fn debug_callback(
    _message_severity: vk::DebugUtilsMessageSeverityFlagBitsEXT,
    _message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    eprintln!("{}", CStr::from_ptr((*p_callback_data).p_message).to_string_lossy());
    vk::FALSE
}

pub fn create_debug_messenger(instance: &InstanceLoader) -> Result<vk::DebugUtilsMessengerEXT> {
    let create_info = vk::DebugUtilsMessengerCreateInfoEXTBuilder::new()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR_EXT |
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING_EXT |
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE_EXT |
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO_EXT
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL_EXT |
            vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE_EXT |
            vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION_EXT
        )
        .pfn_user_callback(Some(debug_callback));

    unsafe { instance.create_debug_utils_messenger_ext(&create_info, None) }
        .result()
        .context("Failed to create debug messenger")
}