use cgmath::Deg;
use cgmath::Matrix4;
use cgmath::Point3;
use cgmath::Rad;
use cgmath::Vector3;
use fuji::Fuji;
use fuji::FujiBuilder;
use png::HasParameters;
use std::collections::HashSet;
use std::fs::File;
use std::iter::FromIterator;
use std::ops::Deref;
use std::sync::Arc;
use std::time::Instant;
use vulkano::SynchronizedVulkanObject;
use vulkano::VulkanObject;
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
use vulkano::image::ImageAccess;
use vulkano::image::ImageUsage;
use vulkano::image::swapchain::SwapchainImage;
use vulkano::impl_vertex;
use vulkano::instance::ApplicationInfo;
use vulkano::instance::Instance;
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

#[derive(Debug)]
struct HelloTriangle {
    fuji: Fuji,

    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    graphics_pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,

    swap_chain_framebuffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,

    vertex_buffer:   Arc<dyn BufferAccess + Send + Sync>,
    index_buffer:    Arc<dyn TypedBufferAccess<Content=[u32]> + Send + Sync>,
    uniform_buffers: Vec<Arc<CpuAccessibleBuffer<UniformBufferObject>>>,
    command_buffers: Vec<Arc<AutoCommandBuffer>>,

    previous_frame_end: Option<Box<GpuFuture>>,
    recreate_swap_chain: bool,

    state: UniformBufferObject,
    start_time: Instant,
}

impl HelloTriangle {
    fn new() -> Self {
        let fuji = FujiBuilder::new()
            .with_window()
            .build()
            .expect("fuji instance")
            .with_graphics_queue()
            .with_present_queue()
            .with_swapchain()
            .build()
            .expect("fuji devices");

        let app = HelloTriangle {
            fuji,
        };

        app
    }

    fn create_sync_objects(device: &Arc<Device>) -> Box<GpuFuture> {
        Box::new(vulkano::sync::now(device.clone())) as Box<GpuFuture>
    }

    fn create_render_pass(device: &Arc<Device>, color_format: Format) -> Arc<dyn RenderPassAbstract + Send + Sync> {
        Arc::new(vulkano::single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: color_format,
                    samples: 1,
                },
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: Format::D32Sfloat,
                    samples: 1,
                }
            },
            pass: {
                color: [ color ],
                depth_stencil: { depth }
            }
        ).unwrap())
    }

    fn create_graphics_pipeline(
        device: &Arc<Device>,
        swap_chain_extent: [u32; 2],
        render_pass: &Arc<dyn RenderPassAbstract + Send + Sync>,
    ) -> Arc<dyn GraphicsPipelineAbstract + Send + Sync> {
        mod vertex_shader {
            vulkano_shaders::shader! {
               ty: "vertex",
               path: "src/shader.vert",
            }
        }

        mod fragment_shader {
            vulkano_shaders::shader! {
                ty: "fragment",
                path: "src/shader.frag",
            }
        }

        let vert_shader_module = vertex_shader::Shader::load(device.clone()).expect("vertex shader module");
        let frag_shader_module = fragment_shader::Shader::load(device.clone()).expect("fragment shader module");

        let dimensions = [ swap_chain_extent[0] as f32, swap_chain_extent[1] as f32 ];
        let viewport = Viewport {
            origin: [ 0.0, 0.0 ],
            dimensions,
            depth_range: 0.0 .. 1.0,
        };

        Arc::new(GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vert_shader_module.main_entry_point(), ())
            .triangle_strip()
            .primitive_restart(true)
            .viewports(vec![ viewport ]) // NOTE: also sets scissor to cover whole viewport
            .fragment_shader(frag_shader_module.main_entry_point(), ())
            .depth_clamp(false)
            // NOTE: there's an outcommented .rasterizer_discard() in Vulkano...
            .polygon_mode_fill()
            .line_width(1.0)
            .cull_mode_back()
            .front_face_clockwise()
            // NOTE: no depth_bias here, but on pipeline::raster::Rasterization
            .blend_pass_through()
            .depth_stencil_simple_depth()
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap())
    }

    fn create_framebuffers(
        device: &Arc<Device>,
        swap_chain_images: &[Arc<SwapchainImage<Window>>],
        render_pass: &Arc<dyn RenderPassAbstract + Send + Sync>
    ) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
        let dimensions = swap_chain_images[0].dimensions();

        swap_chain_images.iter()
            .map(|image| -> Arc<dyn FramebufferAbstract + Send + Sync> {
                let depth_buffer = AttachmentImage::transient(device.clone(), dimensions, Format::D32Sfloat).unwrap();

                Arc::new(Framebuffer::start(render_pass.clone())
                         .add(image.clone()).unwrap()
                         .add(depth_buffer.clone()).unwrap()
                         .build().unwrap())
            })
            .collect::<Vec<_>>()
    }

    fn create_terrain(graphics_queue: &Arc<Queue>, heightmap: Vec<u16>, width: u32, height: u32) -> (Arc<dyn BufferAccess + Send + Sync>, Arc<dyn TypedBufferAccess<Content=[u32]> + Send + Sync>) {
        let vertices = heightmap.iter().enumerate().map(|(idx, height)| {
            let x = ((idx as u32 % width) as f32 / width as f32 - 0.5) * 10.0;
            let y = ((idx as u32 / width) as f32 / width as f32 - 0.5) * 10.0;
            Vertex::new([ x, y, *height as f32 / 65535.0 ], [ 0.1, 0.2, 0.0 ])
        })
        .collect::<Vec<_>>();

        let indices = (1 .. height).flat_map(|y| {
                (0 .. width).flat_map(move |x| {
                    vec![
                        (y - 1) * height + x,
                        (y - 0) * height + x,
                    ]
                })
                .chain([ u32::max_value() ].into_iter().cloned())
            })
            .collect::<Vec<_>>();

        let (vertex_buffer, vertex_future) = ImmutableBuffer::from_iter(
                vertices.iter().cloned(), BufferUsage::vertex_buffer(),
                graphics_queue.clone())
            .unwrap();

        let (index_buffer, index_future) = ImmutableBuffer::from_iter(
                indices.iter().cloned(), BufferUsage::index_buffer(),
                graphics_queue.clone())
            .unwrap();

        vertex_future.flush().unwrap();
        index_future.flush().unwrap();

        (vertex_buffer, index_buffer)
    }

    fn create_state(dimensions: [u32; 2]) -> UniformBufferObject {
        let dimensions = [ dimensions[0] as f32, dimensions[1] as f32 ];

        let model = Matrix4::from_angle_z(Rad(0.0));

        let view = Matrix4::look_at(
            Point3::new(2.0, 2.0, 1.5),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        );

        let mut proj = cgmath::perspective(
            Rad::from(Deg(45.0)),
            dimensions[0] as f32 / dimensions[1] as f32,
            0.1,
            10.0,
        );

        proj.y.y *= -1.0;

        UniformBufferObject {
            model, view, proj,
        }
    }

    fn create_uniform_buffers(device: &Arc<Device>, num_buffers: usize, state: &UniformBufferObject) -> Vec<Arc<CpuAccessibleBuffer<UniformBufferObject>>> {
        let mut buffers = Vec::new();

        for _ in 0..num_buffers {
            let buffer = CpuAccessibleBuffer::from_data(
                device.clone(),
                BufferUsage::uniform_buffer_transfer_destination(),
                state.clone(),
            ).unwrap();

            buffers.push(buffer);
        }

        buffers
    }

    fn update_uniform_buffer(&self, idx: usize) {
        let mut buf = self.uniform_buffers[idx].write().unwrap();
        *buf = self.state.clone();
    }

    fn create_command_buffers(&mut self) {
        let queue_family = self.graphics_queue.family();

        let set = Arc::new(
            PersistentDescriptorSet::start(self.graphics_pipeline.clone(), 0)
                .add_buffer(self.vr_uniform_buffer.clone()).unwrap()
                .build().unwrap()
        );

        self.vr_cmdbuffer = Some(Arc::new(AutoCommandBufferBuilder::primary(self.device.clone(), queue_family).unwrap()
                .begin_render_pass(self.vr_image.clone(), false, vec![ [ 0.392, 0.584, 0.929, 1.0 ].into(), 1.0.into() ]).unwrap()
                .draw_indexed(
                    self.graphics_pipeline.clone(),
                    &DynamicState::none(),
                    vec![ self.vertex_buffer.clone() ],
                    self.index_buffer.clone(),
                    set.clone(),
                    (),
                ).unwrap()
                .end_render_pass().unwrap()
                .build().unwrap()));

        self.command_buffers = self.swap_chain_framebuffers.iter().enumerate()
            .map(|(idx, framebuffer)| {
                let set = Arc::new(
                    PersistentDescriptorSet::start(self.graphics_pipeline.clone(), 0)
                        .add_buffer(self.uniform_buffers[idx].clone()).unwrap()
                        .build().unwrap()
                );

                Arc::new(AutoCommandBufferBuilder::primary_simultaneous_use(self.device.clone(), queue_family).unwrap()
                         .begin_render_pass(framebuffer.clone(), false, vec![ [ 0.392, 0.584, 0.929, 1.0 ].into(), 1.0.into() ]).unwrap()
                         .draw_indexed(
                             self.graphics_pipeline.clone(),
                             &DynamicState::none(),
                             vec![ self.vertex_buffer.clone() ],
                             self.index_buffer.clone(),
                             set.clone(),
                             (),
                         ).unwrap()
                         .end_render_pass().unwrap()
                         .build().unwrap())
            })
            .collect();
    }

    fn main_loop(&mut self) {
        loop {
            let mut done = false;

            self.events_loop.poll_events(|ev| {
                match ev {
                    Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                        done = true;
                    },
                    Event::WindowEvent { event: WindowEvent::KeyboardInput { input, .. }, .. } => {
                        match input.virtual_keycode {
                            Some(VirtualKeyCode::Escape) => {
                                done = true;
                            },
                            _ => ()
                        }
                    },
                    _ => ()
                }
            });

            self.update_state();

            self.draw_frame();

            if done {
                return;
            }
        }
    }

    fn update_state(&mut self) {
        let elapsed = self.start_time.elapsed();
        let elapsed = (elapsed.as_secs() * 1000) + u64::from(elapsed.subsec_millis());

        let model = Matrix4::from_angle_z(Rad::from(Deg(elapsed as f32 * 0.010)));

        self.state.model = model;
    }

    fn draw_frame(&mut self) {
        self.previous_frame_end.as_mut().unwrap().cleanup_finished();

        if self.recreate_swap_chain {
            self.recreate_swap_chain();
            self.recreate_swap_chain = false;
        }

        let (image_index, acquire_future) = match vulkano::swapchain::acquire_next_image(self.swap_chain.clone(), None) {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                self.recreate_swap_chain = true;
                return;
            }
            Err(e) => panic!("{:?}", e),
        };

        self.update_uniform_buffer(image_index);

        let command_buffer = self.command_buffers[image_index].clone();

        let future = acquire_future
            .then_execute(self.graphics_queue.clone(), command_buffer).unwrap()
            .then_swapchain_present(self.present_queue.clone(), self.swap_chain.clone(), image_index)
            .then_signal_fence_and_flush();

        let physical_device   = PhysicalDevice::from_index(&self.instance, self.physical_device_index).unwrap();
        let [ width, height ] = self.vr_image.dimensions().width_height();
        let format            = self.vr_image.format() as u32;
        let sample_count      = self.vr_image.samples();

        let eye = openvr::Eye::Left;
        let handle = openvr::compositor::texture::vulkan::Texture {
            image:              self.vr_image.inner().image.internal_object(),
            device:             self.device.internal_object() as *mut _,
            physical_device:    physical_device.internal_object() as *mut _,
            instance:           self.instance.internal_object() as *mut _,
            queue:              *self.graphics_queue.internal_object_guard().deref() as *mut _,
            queue_family_index: self.graphics_queue.id_within_family(),

            width, height, format, sample_count,
        };
        let texture = openvr::compositor::texture::Texture {
            handle:      openvr::compositor::texture::Handle::Vulkan(handle),
            color_space: openvr::compositor::texture::ColorSpace::Auto,
        };
        unsafe { self.vr.compositor.submit(eye, &texture, None, None).expect("submit") };

        match future {
            Ok(future) => {
                self.previous_frame_end = Some(Box::new(future) as Box<_>);
            },
            Err(vulkano::sync::FlushError::OutOfDate) => {
                self.recreate_swap_chain = true;
                self.previous_frame_end = Some(Box::new(vulkano::sync::now(self.device.clone())) as Box<_>);
            },
            Err(e) => {
                eprintln!("{:?}", e);
                self.previous_frame_end = Some(Box::new(vulkano::sync::now(self.device.clone())) as Box<_>);
            },
        }
    }
}

fn main() {
    let app = HelloTriangle::new();

    dbg!(app);
}
