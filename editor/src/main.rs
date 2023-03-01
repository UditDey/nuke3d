mod cli_args;

use common::{
    window::{create_window, WindowEvent},
    renderer::{Renderer, RendererConfig},
    anyhow::Result
};

use cli_args::CliArgs;

fn main() -> Result<()> {
    let cli_args: CliArgs = argh::from_env();

    let window = create_window(900, 600, "Nuke3D Editor")?;

    let renderer_config = RendererConfig {
        device_name: cli_args.rend_device.as_deref(),
        force_validation: cli_args.rend_validation
    };

    let _renderer = Renderer::new(&renderer_config, window.as_ref())?;

    window.set_visible(true);

    loop {
        let event = window.next_event();

        if let WindowEvent::ShouldClose = event {
            break;
        }
    }

    Ok(())
}