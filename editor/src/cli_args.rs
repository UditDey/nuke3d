use argh::FromArgs;

#[derive(FromArgs, Debug)]
/// The nuke3d visual editor
pub struct CliArgs {
    /// provide a name to override the default vulkan device
    #[argh(option)]
    pub rend_device: Option<String>,

    /// force vulkan validation layers
    #[argh(switch)]
    pub rend_validation: bool,
    
    /// override number of frames in flight
    #[argh(option)]
    pub rend_frames_in_flight: Option<u32>
}