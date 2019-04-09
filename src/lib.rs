use log::debug;
use log::error;
use log::info;
use log::warn;
use std::collections::HashSet;
use std::ffi::CString;
use std::fs::File;
use std::iter::FromIterator;
use std::ops::Deref;
use std::sync::Arc;
use std::time::Instant;
use vulkano::buffer::BufferAccess;
use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::buffer::TypedBufferAccess;
use vulkano::buffer::immutable::ImmutableBuffer;
use vulkano::command_buffer::AutoCommandBuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::DynamicState;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::Device;
use vulkano::device::DeviceExtensions;
use vulkano::device::Features;
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::framebuffer::Framebuffer;
use vulkano::framebuffer::FramebufferAbstract;
use vulkano::framebuffer::RenderPassAbstract;
use vulkano::framebuffer::Subpass;
use vulkano::image::AttachmentImage;
use vulkano::image::ImageUsage;
use vulkano::image::swapchain::SwapchainImage;
use vulkano::impl_vertex;
use vulkano::instance::ApplicationInfo;
use vulkano::instance::Instance;
use vulkano::instance::InstanceCreationError;
use vulkano::instance::PhysicalDevice;
use vulkano::instance::RawInstanceExtensions;
use vulkano::instance::Version;
use vulkano::instance::debug::DebugCallback;
use vulkano::instance::debug::MessageTypes;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::GraphicsPipelineAbstract;
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain::AcquireError;
use vulkano::swapchain::Capabilities;
use vulkano::swapchain::ColorSpace;
use vulkano::swapchain::CompositeAlpha;
use vulkano::swapchain::PresentMode;
use vulkano::swapchain::SupportedPresentModes;
use vulkano::swapchain::Surface;
use vulkano::swapchain::Swapchain;
use vulkano::sync::GpuFuture;
use vulkano::sync::SharingMode;
use vulkano_win::VkSurfaceBuild;
use winit::Event;
use winit::EventsLoop;
use winit::VirtualKeyCode;
use winit::Window;
use winit::WindowBuilder;
use winit::WindowEvent;
use winit::dpi::LogicalSize;
use winit::os::unix::WindowBuilderExt;

const VALIDATION_LAYERS: &[&str] = &[
    "VK_LAYER_LUNARG_standard_validation",
];

#[cfg(not(release))]
const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(release)]
const ENABLE_VALIDATION_LAYERS: bool = false;

#[derive(Debug)]
pub struct FujiBuilder {
    app_info:   Option<ApplicationInfo<'static>>,
    extensions: RawInstanceExtensions,
    layers:     Vec<String>,
}

impl FujiBuilder {
    pub fn new() -> Self {
        Self {
            app_info:   None,
            extensions: RawInstanceExtensions::none(),
            layers:     vec![],
        }
    }

    pub fn with_windowing_extensions(self) -> Self {
        Self {
            extensions: self.extensions.union(&(&vulkano_win::required_extensions()).into()),
            ..self
        }
    }

    pub fn build_instance(mut self) -> Result<FujiStage2Builder, InstanceCreationError> {
        let mut validation_layers = false;

        if ENABLE_VALIDATION_LAYERS {
            if !Self::check_validation_layer_support() {
                warn!("Validation layers requested, but not available!")
            }
            else {
                // TODO: Should be ext_debug_utils but vulkano doesn't support that yet
                self.extensions.insert(CString::new("VK_ext_debug_report").unwrap());
                validation_layers = true;
            }
        }

        let instance = Instance::new(
            self.app_info.as_ref(),
            self.extensions,
            self.layers.iter().map(Deref::deref)
                .chain(if validation_layers { VALIDATION_LAYERS.iter() } else { [].iter() }.cloned()),
        )?;

        if validation_layers {
            Self::setup_debug_callback(&instance);
        }

        Ok(FujiStage2Builder {
            instance,
        })
    }

    fn check_validation_layer_support() -> bool {
        let layers: Vec<_> = vulkano::instance::layers_list().unwrap()
            .map(|l| l.name().to_string()).collect();

        VALIDATION_LAYERS.iter().all(|l| layers.contains(&l.to_string()))
    }

    fn setup_debug_callback(instance: &Arc<Instance>) -> Option<DebugCallback> {
        if !ENABLE_VALIDATION_LAYERS {
            return None;
        }

        let msg_types = MessageTypes {
            error: true,
            warning: true,
            performance_warning: false,
            information: false,
            debug: true,
        };

        DebugCallback::new(&instance, msg_types, |msg| {
            match msg.ty {
                MessageTypes { error: true, .. }               => error!(target: "vulkan", "{}", msg.description),
                MessageTypes { warning: true, .. }             => warn!(target:  "vulkan", "{}", msg.description),
                MessageTypes { performance_warning: true, .. } => warn!(target:  "vulkan", "[perf] {}", msg.description),
                MessageTypes { information: true,  .. }        => info!(target:  "vulkan", "{}", msg.description),
                _                                              => debug!(target: "vulkan", "{}", msg.description),
            }
        }).ok()
    }
}

#[derive(Debug)]
pub struct FujiStage2Builder {
    instance: Arc<Instance>,
}
