mod cli_args;

use std::thread;
use std::sync::Arc;

use parking_lot::RwLock;

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
        force_validation: cli_args.rend_validation,
        frames_in_flight: cli_args.rend_frames_in_flight
    };

    let mut renderer = Renderer::new(&renderer_config, window.as_ref())?;
    let result = Arc::new(RwLock::new(None));
    
    // Start render loop
    let render_loop = thread::spawn({
        let result = result.clone();
        
        move || {
            loop {
                // If result set by other thread, exit
                if result.read().is_some() {
                    break;
                }
                
                let res = renderer.render_frame();
                
                // Time to exit
                if res.is_err() {
                    *result.write() = Some(res);
                    break;
                }
            }
            
            renderer.destroy();
        }
    });

    // Start window event loop    
    window.set_visible(true);

    loop {
        // If result set by other thread, exit
        if result.read().is_some() {
            break;
        }
        
        let event = window.next_event();

        // Time to exit, set result to Ok(())
        if let WindowEvent::ShouldClose = event {
            *result.write() = Some(Ok(()));
            break;
        }
    }
    
    window.set_visible(false);
    
    // Wait for other threads to exit
    render_loop.join().unwrap();
    
    let mut result = result.write();
    result.take().unwrap()
}