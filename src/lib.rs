#![feature(concat_idents, type_ascription, never_type, specialization)]

use log::debug;
use log::error;
use log::info;
use log::warn;
use std::collections::HashSet;
use std::ffi::CString;
use std::iter::FromIterator;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::RwLock;
use vulkano::buffer::BufferAccess;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::DrawError;
use vulkano::command_buffer::DynamicState;
use vulkano::device::Device;
use vulkano::device::DeviceCreationError;
use vulkano::device::DeviceExtensions;
use vulkano::device::Features;
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::framebuffer::Framebuffer;
use vulkano::framebuffer::FramebufferAbstract;
use vulkano::framebuffer::RenderPassAbstract;
use vulkano::image::ImageUsage;
use vulkano::image::swapchain::SwapchainImage;
use vulkano::instance::ApplicationInfo;
use vulkano::instance::Instance;
use vulkano::instance::InstanceCreationError;
use vulkano::instance::PhysicalDevice;
use vulkano::instance::RawInstanceExtensions;
use vulkano::instance::debug::DebugCallback;
use vulkano::instance::debug::MessageTypes;
use vulkano::pipeline::GraphicsPipelineAbstract;
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain::Capabilities;
use vulkano::swapchain::ColorSpace;
use vulkano::swapchain::CompositeAlpha;
use vulkano::swapchain::PresentMode;
use vulkano::swapchain::SupportedPresentModes;
use vulkano::swapchain::Surface;
use vulkano::swapchain::Swapchain;
use vulkano::sync::SharingMode;
use vulkano_win::VkSurfaceBuild;
use winit::EventsLoop;
use winit::Window;
use winit::WindowBuilder;
use winit::dpi::LogicalSize;
use winit::os::unix::WindowBuilderExt;

const WIDTH:  u32 = 1280;
const HEIGHT: u32 = 720;

const VALIDATION_LAYERS: &[&str] = &[
    "VK_LAYER_LUNARG_standard_validation",
];

#[cfg(not(release))]
const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(release)]
const ENABLE_VALIDATION_LAYERS: bool = false;

pub struct PhysicalDeviceFacade(Arc<Instance>, usize);

impl PhysicalDeviceFacade {
    pub fn get<'a>(&'a self) -> PhysicalDevice<'a> {
        PhysicalDevice::from_index(&self.0, self.1).unwrap()
    }
}

impl<'a> From<PhysicalDevice<'a>> for PhysicalDeviceFacade {
    fn from(dev: PhysicalDevice) -> Self {
        PhysicalDeviceFacade(dev.instance().clone(), dev.index())
    }
}

enum Slot<T> {
    Empty,
    Requested,
    Found(T),
}

impl<T> Slot<T> {
    fn is_satisfied(&self) -> bool {
        match self {
            Slot::Empty     => true,
            Slot::Requested => false,
            Slot::Found(_)  => true,
        }
    }

    fn is_requested(&self) -> bool {
        match self {
            Slot::Empty     => false,
            Slot::Requested => true,
            Slot::Found(_)  => true,
        }
    }

    fn mimic<U>(&self) -> Slot<U> {
        match self {
            Slot::Empty     => Slot::Empty,
            Slot::Requested => Slot::Requested,
            Slot::Found(_)  => Slot::Requested,
        }
    }

    fn resolve<F: FnOnce() -> T>(&mut self, f: F) {
        if !self.is_satisfied() {
            *self = Slot::Found(f());
        }
    }

    fn unwrap(self) -> T {
        match self {
            Slot::Empty     => panic!("called `Slot::unwrap()` on an `Empty` value"),
            Slot::Requested => panic!("called `Slot::unwrap()` on a `Requested` value"),
            Slot::Found(i)  => i,
        }
    }

    fn map<P: FnOnce() -> R, R>(self, f: P) -> Option<R> {
        if let Slot::Requested = self {
            Some(f())
        }
        else {
            None
        }
    }
}

impl<T: Clone> Slot<T> {
    fn cloned(&self) -> Self {
        match self {
            Slot::Empty     => Slot::Empty,
            Slot::Requested => Slot::Requested,
            Slot::Found(t)  => Slot::Found(t.clone()),
        }
    }
}

impl<T> Into<Option<T>> for Slot<T> {
    fn into(self) -> Option<T> {
        match self {
            Slot::Empty     => None,
            Slot::Requested => None,
            Slot::Found(i)  => Some(i),
        }
    }
}

struct QueueFamilyIndices {
    graphics_family: Slot<u32>,
    present_family:  Slot<u32>,
}

impl QueueFamilyIndices {
    fn new() -> Self {
        QueueFamilyIndices {
            graphics_family: Slot::Empty,
            present_family:  Slot::Empty,
        }
    }

    fn is_complete(&self) -> bool {
        self.graphics_family.is_satisfied() && self.present_family.is_satisfied()
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = u32> {
        vec![
            self.graphics_family.cloned().into(),
            self.present_family.cloned().into(): Option<u32>
        ].into_iter().flatten()
    }
}

pub struct FujiBuilder {
    app_info:   Option<ApplicationInfo<'static>>,
    extensions: RawInstanceExtensions,
    layers:     Vec<String>,

    window_handles: Slot<(EventsLoop, Arc<Surface<Window>>)>,
}

impl FujiBuilder {
    pub fn new() -> Self {
        Self {
            app_info:   None,
            extensions: RawInstanceExtensions::none(),
            layers:     vec![],

            window_handles: Slot::Empty,
        }
    }

    pub fn with_window(self) -> Self {
        Self {
            extensions:     self.extensions.union(&(&vulkano_win::required_extensions()).into()),
            window_handles: Slot::Requested,
            ..self
        }
    }

    pub fn build(mut self) -> Result<FujiDeviceBuilder, InstanceCreationError> {
        let mut validation_layers = false;

        if ENABLE_VALIDATION_LAYERS {
            if !Self::check_validation_layer_support() {
                warn!("Validation layers requested, but not available!")
            }
            else {
                // TODO: Should be ext_debug_utils but vulkano doesn't support that yet
                self.extensions.insert(CString::new("VK_EXT_debug_report").unwrap());
                validation_layers = true;
            }
        }

        let instance = Instance::new(
            self.app_info.as_ref(),
            self.extensions,
            self.layers.iter().map(Deref::deref)
                .chain(if validation_layers { VALIDATION_LAYERS.iter() } else { [].iter() }.cloned()),
        )?;

        let debug_callback = if validation_layers {
            Self::setup_debug_callback(&instance)
        }
        else {
            None
        };

        let window_handles = self.window_handles.map(|| Self::create_surface(&instance));

        Ok(FujiDeviceBuilder {
            instance, window_handles,
            debug_callback,

            graphics_queue: Slot::Empty,
            present_queue:  Slot::Empty,
            swapchain:      Slot::Empty,
        })
    }

    fn create_surface(instance: &Arc<Instance>) -> (EventsLoop, Arc<Surface<Window>>) {
        let events_loop = EventsLoop::new();
        let surface = WindowBuilder::new()
            .with_class("vulkan".to_string(), "vulkan".to_string())
            .with_title("Vulkan")
            .with_dimensions(LogicalSize::new(WIDTH as f64, HEIGHT as f64));

        let surface = surface
            .build_vk_surface(&events_loop, instance.clone())
            .expect("window surface");

        (events_loop, surface)
    }

    fn check_validation_layer_support() -> bool {
        let layers: Vec<_> = vulkano::instance::layers_list().unwrap()
            .map(|l| l.name().to_string()).collect();

        VALIDATION_LAYERS.iter().all(|l| layers.contains(&l.to_string()))
    }

    fn setup_debug_callback(instance: &Arc<Instance>) -> Option<DebugCallback> {
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

pub struct FujiDeviceBuilder {
    instance:       Arc<Instance>,
    window_handles: Option<(EventsLoop, Arc<Surface<Window>>)>,
    debug_callback: Option<DebugCallback>,

    graphics_queue: Slot<Arc<Queue>>,
    present_queue:  Slot<Arc<Queue>>,
    swapchain:      Slot<(Swapchain<Window>, Vec<SwapchainImage<Window>>)>,
}

impl FujiDeviceBuilder {
    pub fn with_graphics_queue(self) -> Self {
        Self {
            graphics_queue: Slot::Requested,
            ..self
        }
    }

    pub fn with_present_queue(self) -> Self {
        Self {
            present_queue: Slot::Requested,
            ..self
        }
    }

    pub fn with_swapchain(self) -> Self {
        Self {
            swapchain: Slot::Requested,
            ..self
        }
    }

    pub fn build(mut self) -> Result<Arc<Fuji>, DeviceCreationError> {
        let physical_device = self.pick_physical_device();
        let device          = self.create_logical_device(physical_device.get())?;

        let (events_loop, surface) = self.window_handles
            .map(|(e, s)| (Some(e), Some(s)))
            .unwrap_or((None, None));

        let surface_format = surface.as_ref().map(|surface| {
            let capabilities = surface.capabilities(physical_device.get()).expect("surface capabilities");
            Self::choose_swap_surface_format(&capabilities.supported_formats)
        });

        let events_loop = RwLock::new(events_loop);

        let fuji = Fuji {
            instance:       self.instance,
            debug_callback: self.debug_callback,

            events_loop, surface,

            physical_device, device,

            graphics_queue: self.graphics_queue.into(),
            present_queue:  self.present_queue.into(),

            surface_format,
            swapchain: RwLock::new(None),
        };

        Ok(fuji.into())
    }

    fn pick_physical_device(&self) -> PhysicalDeviceFacade {
        PhysicalDevice::enumerate(&self.instance)
            .find(|device| self.is_device_suitable(*device))
            .expect("suitable GPU")
            .into()
    }

    fn is_device_suitable(&self, device: PhysicalDevice) -> bool {
        if !self.find_queue_families(device).is_complete() {
            return false
        }

        if !self.check_device_extension_support(device) {
            return false
        }

        if let Some((_, ref surface)) = self.window_handles {
            let capabilities = surface.capabilities(device).expect("surface capabilities");

            return !capabilities.supported_formats.is_empty() && capabilities.present_modes.iter().next().is_some()
        }

        true
    }

    fn device_extensions(&self) -> DeviceExtensions {
        let mut device_extensions = vulkano::device::DeviceExtensions::none();

        if self.swapchain.is_requested() {
            device_extensions.khr_swapchain = true;
        }

        device_extensions
    }

    fn check_device_extension_support(&self, device: PhysicalDevice) -> bool {
        let available_extensions = DeviceExtensions::supported_by_device(device);
        let device_extensions = self.device_extensions();

        available_extensions.intersection(&device_extensions) == device_extensions
    }

    fn choose_swap_surface_format(available_formats: &[ (Format, ColorSpace) ]) -> Format {
        // NOTE: the 'preferred format' mentioned in the tutorial doesn't seem to be
        // queryable in Vulkano (no VK_FORMAT_UNDEFINED enum)

        *available_formats.iter().find(|(format, color_space)| {
            *format == Format::B8G8R8A8Unorm && *color_space == ColorSpace::SrgbNonLinear
        })
        .map(|(format, _color_space)| format)
        .unwrap_or_else(|| &available_formats[0].0)
    }

    fn find_queue_families(&self, device: PhysicalDevice) -> QueueFamilyIndices {
        let mut indices = QueueFamilyIndices::new();

        indices.graphics_family = self.graphics_queue.mimic();
        indices.present_family  = self.present_queue.mimic();

        for (id, queue_family) in device.queue_families().enumerate() {
            if queue_family.supports_graphics() {
                indices.graphics_family.resolve(|| id as u32);
            }

            if let Some((_, ref surface)) = self.window_handles {
                if surface.is_supported(queue_family).unwrap() {
                    indices.present_family.resolve(|| id as u32);
                }
            }

            if indices.is_complete() {
                break;
            }
        }

        indices
    }

    fn create_logical_device(&mut self, physical_device: PhysicalDevice) -> Result<Arc<Device>, DeviceCreationError> {
        let indices = self.find_queue_families(physical_device);
        let unique_queue_families: HashSet<u32> = HashSet::from_iter(indices.iter());

        let queue_priority = 1.0;
        let queue_families = unique_queue_families.iter().map(|i| {
            (physical_device.queue_families().nth(*i as usize).unwrap(), queue_priority)
        });

        let (device, queues) = Device::new(
            physical_device,
            &Features::none(),
            &self.device_extensions(),
            queue_families
        )?;

        let queues: Vec<_> = queues.collect();

        self.graphics_queue.resolve(|| queues.iter().find(|q| q.family().id() == indices.graphics_family.cloned().unwrap()).unwrap().clone());
        self.present_queue.resolve(||  queues.iter().find(|q| q.family().id() == indices.present_family.cloned().unwrap()).unwrap().clone());

        Ok(device)
    }
}

pub struct Fuji {
    instance:       Arc<Instance>,
    debug_callback: Option<DebugCallback>,

    events_loop: RwLock<Option<EventsLoop>>,
    surface:     Option<Arc<Surface<Window>>>,

    physical_device: PhysicalDeviceFacade,
    device:          Arc<Device>,

    graphics_queue: Option<Arc<Queue>>,
    present_queue:  Option<Arc<Queue>>,

    surface_format: Option<Format>,
    swapchain:      RwLock<Option<Arc<SwapchainBundle>>>,
}

pub struct SwapchainBundle {
    pub swapchain:    Arc<Swapchain<Window>>,
    pub images:       Vec<Arc<SwapchainImage<Window>>>,
    pub framebuffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,
    pub viewport:     Viewport,
}

macro_rules! getters {
    ( $field:ident : $type:ty , $($extra:tt)* ) => {
        pub fn $field(&self) -> &$type {
            &self.$field
        }

        getters!{ $($extra)* }
    };

    ( $field:ident : opt $type:ty , $($extra:tt)* ) => {
        pub fn $field(&self) -> &$type {
            &self.$field.as_ref().unwrap()
        }

        getters!{ $($extra)* }
    };

    // `gen` for generation
    //
    // A `RwLock<Option<Arc<T>>>` that we only mutate ourselves by replacing it with a new Arc<T>, and
    // only give out clones of the `Arc<T>`.
    ( $field:ident : opt gen $type:ty , $($extra:tt)* ) => {
        pub fn $field(&self) -> $type {
            self.$field.read().unwrap().as_ref().unwrap().clone()
        }

        getters!{ $($extra)* }
    };

    () => {};
}

impl Fuji {
    getters! {
        instance:        Arc<Instance>,
        debug_callback:  opt DebugCallback,
        surface:         opt Arc<Surface<Window>>,
        physical_device: PhysicalDeviceFacade,
        device:          Arc<Device>,
        graphics_queue:  opt Arc<Queue>,
        present_queue:   opt Arc<Queue>,
        surface_format:  opt Format,
        swapchain:       opt gen Arc<SwapchainBundle>,
    }

    pub fn take_events_loop(&self) -> Option<EventsLoop> {
        self.events_loop.write().unwrap().take()
    }

    pub fn create_swapchain(&self, render_pass: &Arc<dyn RenderPassAbstract + Send + Sync>) {
        let capabilities = self.surface().capabilities(self.physical_device.get()).expect("surface capabilities");
        let present_mode = Self::choose_swap_present_mode(capabilities.present_modes);
        let extent       = Self::choose_swap_extent(&capabilities);

        let mut image_count = capabilities.min_image_count + 1;
        if capabilities.max_image_count.map(|max| image_count > max).unwrap_or(false) {
            image_count = capabilities.max_image_count.unwrap();
        }

        let image_usage = ImageUsage {
            color_attachment: true,
            ..ImageUsage::none()
        };

        let sharing: SharingMode = if self.graphics_queue.as_ref().unwrap().family().id() != self.present_queue.as_ref().unwrap().family().id() {
            vec![
                self.graphics_queue.as_ref().unwrap(),
                self.present_queue.as_ref().unwrap(),
            ].as_slice().into()
        } else {
            self.graphics_queue.as_ref().unwrap().into()
        };

        let old_swapchain_lock = self.swapchain.read().unwrap();
        let old_swapchain = old_swapchain_lock.as_ref().map(|s| &s.swapchain);

        let (swapchain, images) = Swapchain::new(
            self.device.clone(),
            self.surface.as_ref().unwrap().clone(),
            image_count,
            self.surface_format.unwrap(), // TODO: color space?
            extent,
            1, // layers
            image_usage,
            sharing,
            capabilities.current_transform,
            CompositeAlpha::Opaque,
            present_mode,
            true, // clipped
            old_swapchain,
        ).expect("swap chain");

        drop(old_swapchain_lock);

        let framebuffers = images.iter()
            .map(|image| -> Arc<dyn FramebufferAbstract + Send + Sync> {
                Arc::new(Framebuffer::start(render_pass.clone())
                         .add(image.clone()).unwrap()
                         .build().unwrap())
            })
            .collect::<Vec<_>>();

        let dimensions = images[0].dimensions();
        let dimensions = [ dimensions[0] as f32, dimensions[1] as f32 ];
        let viewport = Viewport {
            origin: [ 0.0, 0.0 ],
            dimensions,
            depth_range: 0.0 .. 1.0,
        };

        let mut bundle = self.swapchain.write().unwrap();
        *bundle = Some(Arc::new(SwapchainBundle {
            swapchain, images, framebuffers, viewport,
        }));
    }

    fn choose_swap_present_mode(available_present_modes: SupportedPresentModes) -> PresentMode {
        if available_present_modes.mailbox {
            PresentMode::Mailbox
        } else if available_present_modes.immediate {
            PresentMode::Immediate
        } else {
            PresentMode::Fifo
        }
    }

    fn choose_swap_extent(capabilities: &Capabilities) -> [u32; 2] {
        if let Some(current_extent) = capabilities.current_extent {
            current_extent
        } else {
            let mut actual_extent = [ WIDTH, HEIGHT ];
            actual_extent[0] = capabilities.min_image_extent[0]
                          .max(capabilities.max_image_extent[0].min(actual_extent[0]));
            actual_extent[1] = capabilities.min_image_extent[1]
                          .max(capabilities.max_image_extent[1].min(actual_extent[1]));
            actual_extent
        }
    }
}

pub trait Drawable {
    fn pipeline(&self) -> Arc<dyn GraphicsPipelineAbstract + Send + Sync + 'static>;
    fn dynamic_state(&self) -> DynamicState;
    fn vertex_buffers(&self) -> Vec<Arc<dyn BufferAccess + Send + Sync + 'static>>;
    fn descriptor_sets(&self) -> ();
    fn push_constants(&self) -> ();
}

pub trait CommandBufferBuilderExt: Sized {
    fn fuji_draw<T: Drawable>(self, obj: &T) -> Result<Self, DrawError>;
}

impl CommandBufferBuilderExt for AutoCommandBufferBuilder {
    fn fuji_draw<T: Drawable>(self, obj: &T) -> Result<Self, DrawError> {
        self.draw(
            obj.pipeline(),
            &obj.dynamic_state(),
            obj.vertex_buffers(),
            obj.descriptor_sets(),
            obj.push_constants(),
        )
    }
}
