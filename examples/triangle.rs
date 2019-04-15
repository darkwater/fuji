use fuji::Fuji;
use fuji::FujiBuilder;
use std::sync::Arc;
use vulkano::buffer::BufferAccess;
use vulkano::buffer::BufferUsage;
use vulkano::buffer::immutable::ImmutableBuffer;
use vulkano::command_buffer::AutoCommandBuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::DynamicState;
use vulkano::device::Device;
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::framebuffer::Framebuffer;
use vulkano::framebuffer::FramebufferAbstract;
use vulkano::framebuffer::RenderPassAbstract;
use vulkano::framebuffer::Subpass;
use vulkano::image::ImageAccess;
use vulkano::image::swapchain::SwapchainImage;
use vulkano::impl_vertex;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::GraphicsPipelineAbstract;
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain::AcquireError;
use vulkano::sync::GpuFuture;
use winit::Event;
use winit::VirtualKeyCode;
use winit::Window;
use winit::WindowEvent;

#[derive(Copy, Clone, Debug)]
struct Vertex {
    pos:   [f32; 2],
    color: [f32; 3],
}

impl_vertex!(Vertex, pos, color);
impl Vertex {
    fn new(pos: [f32; 2], color: [f32; 3]) -> Self {
        Self {
            pos, color,
        }
    }
}

struct HelloTriangle {
    fuji: Fuji,

    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    graphics_pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,

    swapchain_framebuffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,

    vertex_buffer:   Arc<dyn BufferAccess + Send + Sync>,
    command_buffers: Vec<Arc<AutoCommandBuffer>>,

    previous_frame_end: Option<Box<GpuFuture>>,
    recreate_swapchain: bool,
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

        let render_pass       = Self::create_render_pass(fuji.device(), fuji.swapchain().format());
        let graphics_pipeline = Self::create_graphics_pipeline(fuji.device(), fuji.swapchain().dimensions(), &render_pass);

        let swapchain_framebuffers = Self::create_framebuffers(fuji.device(), fuji.swapchain_images(), &render_pass);

        let vertex_buffer = Self::create_triangle(fuji.graphics_queue());

        let previous_frame_end = Some(Self::create_sync_objects(fuji.device()));

        let mut app = HelloTriangle {
            fuji,

            render_pass,
            graphics_pipeline,

            swapchain_framebuffers,

            vertex_buffer,
            command_buffers: vec![],

            previous_frame_end,
            recreate_swapchain: false,
        };

        app.create_command_buffers();

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
                }
            },
            pass: {
                color: [ color ],
                depth_stencil: {}
            }
        ).unwrap())
    }

    fn create_graphics_pipeline(
        device: &Arc<Device>,
        swapchain_extent: [u32; 2],
        render_pass: &Arc<dyn RenderPassAbstract + Send + Sync>,
    ) -> Arc<dyn GraphicsPipelineAbstract + Send + Sync> {
        mod vertex_shader {
            vulkano_shaders::shader! {
               ty: "vertex",
               path: "examples/triangle.vert",
            }
        }

        mod fragment_shader {
            vulkano_shaders::shader! {
                ty: "fragment",
                path: "examples/triangle.frag",
            }
        }

        let vert_shader_module = vertex_shader::Shader::load(device.clone()).expect("vertex shader module");
        let frag_shader_module = fragment_shader::Shader::load(device.clone()).expect("fragment shader module");

        let dimensions = [ swapchain_extent[0] as f32, swapchain_extent[1] as f32 ];
        let viewport = Viewport {
            origin: [ 0.0, 0.0 ],
            dimensions,
            depth_range: 0.0 .. 1.0,
        };

        Arc::new(GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vert_shader_module.main_entry_point(), ())
            .triangle_list()
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
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap())
    }

    fn create_framebuffers(
        device: &Arc<Device>,
        swapchain_images: &[Arc<SwapchainImage<Window>>],
        render_pass: &Arc<dyn RenderPassAbstract + Send + Sync>
    ) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
        let dimensions = swapchain_images[0].dimensions().width_height();

        swapchain_images.iter()
            .map(|image| -> Arc<dyn FramebufferAbstract + Send + Sync> {
                Arc::new(Framebuffer::start(render_pass.clone())
                         .add(image.clone()).unwrap()
                         .build().unwrap())
            })
            .collect::<Vec<_>>()
    }

    fn create_triangle(graphics_queue: &Arc<Queue>) -> Arc<dyn BufferAccess + Send + Sync> {
        let vertices = [
            Vertex::new([  0.0, -0.5 ], [ 0.2, 0.3, 0.8 ]),
            Vertex::new([  0.5,  0.5 ], [ 0.1, 0.2, 0.8 ]),
            Vertex::new([ -0.5,  0.5 ], [ 0.1, 0.2, 0.7 ]),
        ];

        let (vertex_buffer, vertex_future) = ImmutableBuffer::from_iter(
                vertices.iter().cloned(), BufferUsage::vertex_buffer(),
                graphics_queue.clone())
            .unwrap();

        vertex_future.flush().unwrap();

        vertex_buffer
    }

    fn create_command_buffers(&mut self) {
        let queue_family = self.fuji.graphics_queue().family();

        self.command_buffers = self.swapchain_framebuffers.iter().enumerate()
            .map(|(idx, framebuffer)| {
                Arc::new(AutoCommandBufferBuilder::primary_simultaneous_use(self.fuji.device().clone(), queue_family).unwrap()
                         .begin_render_pass(framebuffer.clone(), false, vec![ [ 0.1, 0.1, 0.1, 1.0 ].into() ]).unwrap()
                         .draw(
                             self.graphics_pipeline.clone(),
                             &DynamicState::none(),
                             vec![ self.vertex_buffer.clone() ],
                             (),
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

            self.fuji.events_loop_mut().poll_events(|ev| {
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

            self.draw_frame();

            if done {
                return;
            }
        }
    }

    fn draw_frame(&mut self) {
        self.previous_frame_end.as_mut().unwrap().cleanup_finished();

        // if self.recreate_swapchain {
        //     self.recreate_swapchain();
        //     self.recreate_swapchain = false;
        // }

        let (image_index, acquire_future) = match vulkano::swapchain::acquire_next_image(self.fuji.swapchain().clone(), None) {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                self.recreate_swapchain = true;
                return;
            }
            Err(e) => panic!("{:?}", e),
        };

        let command_buffer = self.command_buffers[image_index].clone();

        let future = acquire_future
            .then_execute(self.fuji.graphics_queue().clone(), command_buffer).unwrap()
            .then_swapchain_present(self.fuji.present_queue().clone(), self.fuji.swapchain().clone(), image_index)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                self.previous_frame_end = Some(Box::new(future) as Box<_>);
            },
            Err(vulkano::sync::FlushError::OutOfDate) => {
                self.recreate_swapchain = true;
                self.previous_frame_end = Some(Box::new(vulkano::sync::now(self.fuji.device().clone())) as Box<_>);
            },
            Err(e) => {
                eprintln!("{:?}", e);
                self.previous_frame_end = Some(Box::new(vulkano::sync::now(self.fuji.device().clone())) as Box<_>);
            },
        }
    }
}

fn main() {
    let mut app = HelloTriangle::new();
    app.main_loop();
}
