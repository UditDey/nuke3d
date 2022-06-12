use std::ffi::CString;

use erupt::{vk, DeviceLoader};
use anyhow::{Result, Context};

pub fn name_object(
    device: &DeviceLoader,
    obj_handle: u64,
    obj_type: vk::ObjectType,
    name: &str
) -> Result<()> {
    if cfg!(debug_assertions) {
        let c_name = CString::new(name).unwrap();

        let name_info = vk::DebugUtilsObjectNameInfoEXTBuilder::new()
            .object_type(obj_type)
            .object_handle(obj_handle)
            .object_name(&c_name);

        unsafe { device.set_debug_utils_object_name_ext(&name_info) }
            .result()
            .context(format!("Failed to set object name \"{name}\""))
    }
    else {
        Ok(())
    }
}

macro_rules! name_multiple {
    ($device:expr, $objects_iter:expr, $object_type:expr, $name:literal) => {
        for (i, obj) in $objects_iter.enumerate() {
            use crate::vk_util::name_object;

            name_object(
                $device,
                obj.object_handle(),
                $object_type,
                &format!(concat!($name, " {}"), i)
            )?;
        }
    };
}

pub(crate) use name_multiple;