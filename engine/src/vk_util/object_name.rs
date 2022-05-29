use std::ffi::CString;

use erupt::{vk, DeviceLoader};
use anyhow::{Result, Context};

pub fn name_object(
    device: &DeviceLoader,
    obj_type: vk::ObjectType,
    obj_handle: u64,
    name: &str
) -> Result<()> {
    if cfg!(debug_assertions) {
        let name = CString::new(name).unwrap();

        let name_info = vk::DebugUtilsObjectNameInfoEXTBuilder::new()
            .object_type(obj_type)
            .object_handle(obj_handle)
            .object_name(&name);

        unsafe { device.set_debug_utils_object_name_ext(&name_info) }
            .result()
            .context("Failed to set object name")
    }
    else {
        Ok(())
    }
}

macro_rules! name_multiple {
    ($device:expr, $objects:expr, $object_type:expr, $name:literal) => {
        for (i, obj) in $objects.iter().enumerate() {
            use crate::vk_util::name_object;

            name_object(
                $device,
                $object_type,
                obj.object_handle(),
                &format!(concat!($name, " {}"), i)
            )?;
        }
    };
}

pub(crate) use name_multiple;