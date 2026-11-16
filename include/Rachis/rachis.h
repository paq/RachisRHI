#ifndef RACHIS_C_ABI_H
#define RACHIS_C_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#ifdef RACHIS_C_EXPORTS
#define RACHIS_API __declspec(dllexport)
#else
#define RACHIS_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) || defined(__clang__)
#define RACHIS_API __attribute__((visibility("default")))
#else
#define RACHIS_API
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#define RACHIS_ABI_VERSION 2U
#define RACHIS_ERROR_MESSAGE_CAPACITY 256u
#define RACHIS_MAX_SURFACE_FORMATS 16U
#define RACHIS_MAX_VERTEX_BUFFERS 8U
#define RACHIS_MAX_VERTEX_ATTRIBUTES 16U

#define RACHIS_REMAINING_SUBRESOURCES 0xffffffffu

#define RACHIS_WHOLE_SIZE 0xffffffffffffffffull

    typedef struct rachis_device rachis_device;
    typedef uint64_t rachis_handle;
    typedef uint32_t rachis_status;

    enum
    {
        RACHIS_STATUS_OK = 0,
        RACHIS_STATUS_INVALID_ARGUMENT = 1,
        RACHIS_STATUS_UNSUPPORTED = 2,
        RACHIS_STATUS_BACKEND_UNAVAILABLE = 3,
        RACHIS_STATUS_ADAPTER_NOT_FOUND = 4,
        RACHIS_STATUS_DEVICE_CREATION_FAILED = 5,
        RACHIS_STATUS_RESOURCE_EXHAUSTED = 6,
        RACHIS_STATUS_INVALID_STATE = 7,
        RACHIS_STATUS_DEVICE_LOST = 8,
        RACHIS_STATUS_OUT_OF_MEMORY = 9,
        RACHIS_STATUS_NATIVE_FAILURE = 10,
        RACHIS_STATUS_VALIDATION_ERROR = 11,
        RACHIS_STATUS_PIPELINE_CACHE_MISS = 12,
        RACHIS_STATUS_NOT_READY = 13,
        RACHIS_STATUS_ABI_MISMATCH = 14
    };

    enum
    {
        RACHIS_BACKEND_D3D12 = 0,
        RACHIS_BACKEND_VULKAN = 1,
        RACHIS_BACKEND_METAL = 2
    };
    enum
    {
        RACHIS_ADAPTER_HIGH_PERFORMANCE = 0,
        RACHIS_ADAPTER_MINIMUM_POWER = 1,
        RACHIS_ADAPTER_WARP = 2
    };
    enum
    {
        RACHIS_ADAPTER_TYPE_UNKNOWN = 0,
        RACHIS_ADAPTER_TYPE_INTEGRATED = 1,
        RACHIS_ADAPTER_TYPE_DISCRETE = 2,
        RACHIS_ADAPTER_TYPE_SOFTWARE = 3
    };
    enum
    {
        RACHIS_VALIDATION_DISABLED = 0,
        RACHIS_VALIDATION_STANDARD = 1,
        RACHIS_VALIDATION_GPU = 2
    };

    enum
    {
        RACHIS_DIAGNOSTIC_INFORMATION = 0,
        RACHIS_DIAGNOSTIC_WARNING = 1,
        RACHIS_DIAGNOSTIC_ERROR = 2,
        RACHIS_DIAGNOSTIC_CORRUPTION = 3
    };
    enum
    {
        RACHIS_QUEUE_GRAPHICS = 0,
        RACHIS_QUEUE_COMPUTE = 1,
        RACHIS_QUEUE_COPY = 2
    };
    enum
    {
        RACHIS_MEMORY_DEVICE_LOCAL = 0,
        RACHIS_MEMORY_UPLOAD = 1,
        RACHIS_MEMORY_READBACK = 2
    };
    enum
    {
        RACHIS_BUFFER_COPY_SOURCE = 1U << 0U,
        RACHIS_BUFFER_COPY_DESTINATION = 1U << 1U,
        RACHIS_BUFFER_VERTEX = 1U << 2U,
        RACHIS_BUFFER_INDEX = 1U << 3U,
        RACHIS_BUFFER_CONSTANT = 1U << 4U,
        RACHIS_BUFFER_STORAGE = 1U << 5U,
        RACHIS_BUFFER_INDIRECT = 1U << 6U
    };
    enum
    {
        RACHIS_SURFACE_WIN32 = 0,
        RACHIS_SURFACE_WAYLAND = 1,
        RACHIS_SURFACE_XCB = 2,
        RACHIS_SURFACE_CA_METAL_LAYER = 3
    };
    enum
    {
        RACHIS_FORMAT_UNKNOWN = 0,
        RACHIS_FORMAT_R8_UNORM = 1,
        RACHIS_FORMAT_RG8_UNORM = 2,
        RACHIS_FORMAT_RGBA8_UNORM = 3,
        RACHIS_FORMAT_RGBA8_UNORM_SRGB = 4,
        RACHIS_FORMAT_BGRA8_UNORM = 5,
        RACHIS_FORMAT_BGRA8_UNORM_SRGB = 6,
        RACHIS_FORMAT_R16_FLOAT = 7,
        RACHIS_FORMAT_RG16_FLOAT = 8,
        RACHIS_FORMAT_RGBA16_FLOAT = 9,
        RACHIS_FORMAT_R32_FLOAT = 10,
        RACHIS_FORMAT_RG32_FLOAT = 11,
        RACHIS_FORMAT_RGBA32_FLOAT = 12,
        RACHIS_FORMAT_D32_FLOAT = 13,
        RACHIS_FORMAT_D24_UNORM_S8_UINT = 14,
        RACHIS_FORMAT_R32_UINT = 22
    };
    enum
    {
        RACHIS_COLOR_SPACE_SRGB_NONLINEAR = 0,
        RACHIS_COLOR_SPACE_EXTENDED_SRGB_LINEAR = 1,
        RACHIS_COLOR_SPACE_HDR10_ST2084 = 2
    };

    enum
    {
        RACHIS_PRESENT_FIFO = 0,
        RACHIS_PRESENT_IMMEDIATE = 1
    };
    enum
    {
        RACHIS_PRESENT_MODE_FIFO_BIT = 1U << 0U,
        RACHIS_PRESENT_MODE_IMMEDIATE_BIT = 1U << 1U
    };

    enum
    {
        RACHIS_PRESENT_STATUS_PRESENTED = 0,
        RACHIS_PRESENT_STATUS_OCCLUDED = 1,
        RACHIS_PRESENT_STATUS_SURFACE_OUT_OF_DATE = 2
    };
    enum
    {
        RACHIS_TEXTURE_1D = 0,
        RACHIS_TEXTURE_2D = 1,
        RACHIS_TEXTURE_3D = 2
    };
    enum
    {
        RACHIS_TEXTURE_COPY_SOURCE = 1U << 0U,
        RACHIS_TEXTURE_COPY_DESTINATION = 1U << 1U,
        RACHIS_TEXTURE_SAMPLED = 1U << 2U,
        RACHIS_TEXTURE_STORAGE = 1U << 3U,
        RACHIS_TEXTURE_RENDER_TARGET = 1U << 4U,
        RACHIS_TEXTURE_DEPTH_STENCIL = 1U << 5U
    };
    enum
    {
        RACHIS_VIEW_USAGE_SAMPLED = 0,
        RACHIS_VIEW_USAGE_STORAGE = 1,
        RACHIS_VIEW_USAGE_RENDER_TARGET = 2,
        RACHIS_VIEW_USAGE_DEPTH_STENCIL = 3
    };
    enum
    {
        RACHIS_VIEW_1D = 0,
        RACHIS_VIEW_1D_ARRAY = 1,
        RACHIS_VIEW_2D = 2,
        RACHIS_VIEW_2D_ARRAY = 3,
        RACHIS_VIEW_3D = 4,
        RACHIS_VIEW_CUBE = 5,
        RACHIS_VIEW_CUBE_ARRAY = 6
    };
    enum
    {
        RACHIS_ASPECT_COLOR = 0,
        RACHIS_ASPECT_DEPTH = 1,
        RACHIS_ASPECT_STENCIL = 2
    };
    enum
    {
        RACHIS_LAYOUT_UNDEFINED = 0,
        RACHIS_LAYOUT_COMMON = 1,
        RACHIS_LAYOUT_PRESENT = 2,
        RACHIS_LAYOUT_SHADER_RESOURCE = 3,
        RACHIS_LAYOUT_UNORDERED_ACCESS = 4,
        RACHIS_LAYOUT_RENDER_TARGET = 5,
        RACHIS_LAYOUT_DEPTH_STENCIL_WRITE = 6,
        RACHIS_LAYOUT_DEPTH_STENCIL_READ = 7,
        RACHIS_LAYOUT_COPY_SOURCE = 8,
        RACHIS_LAYOUT_COPY_DESTINATION = 9,
        RACHIS_LAYOUT_RESOLVE_SOURCE = 10,
        RACHIS_LAYOUT_RESOLVE_DESTINATION = 11
    };
    enum
    {
        RACHIS_STAGE_NONE = 0,
        RACHIS_STAGE_ALL = 1U << 0U,
        RACHIS_STAGE_DRAW = 1U << 1U,
        RACHIS_STAGE_VERTEX_INPUT = 1U << 2U,
        RACHIS_STAGE_VERTEX_SHADER = 1U << 3U,
        RACHIS_STAGE_PIXEL_SHADER = 1U << 4U,
        RACHIS_STAGE_COMPUTE_SHADER = 1U << 5U,
        RACHIS_STAGE_COPY = 1U << 6U,
        RACHIS_STAGE_RESOLVE = 1U << 7U,
        RACHIS_STAGE_RENDER_TARGET = 1U << 8U,
        RACHIS_STAGE_DEPTH_STENCIL = 1U << 9U,
        RACHIS_STAGE_INDIRECT = 1U << 10U
    };
    enum
    {
        RACHIS_ACCESS_NONE = 0,
        RACHIS_ACCESS_COMMON = 1U << 0U,
        RACHIS_ACCESS_VERTEX_BUFFER = 1U << 1U,
        RACHIS_ACCESS_CONSTANT_BUFFER = 1U << 2U,
        RACHIS_ACCESS_INDEX_BUFFER = 1U << 3U,
        RACHIS_ACCESS_SHADER_RESOURCE = 1U << 4U,
        RACHIS_ACCESS_UNORDERED_ACCESS = 1U << 5U,
        RACHIS_ACCESS_COPY_SOURCE = 1U << 6U,
        RACHIS_ACCESS_COPY_DESTINATION = 1U << 7U,
        RACHIS_ACCESS_INDIRECT_ARGUMENT = 1U << 8U,
        RACHIS_ACCESS_RENDER_TARGET = 1U << 9U,
        RACHIS_ACCESS_DEPTH_STENCIL_READ = 1U << 10U,
        RACHIS_ACCESS_DEPTH_STENCIL_WRITE = 1U << 11U,
        RACHIS_ACCESS_RESOLVE_SOURCE = 1U << 12U,
        RACHIS_ACCESS_RESOLVE_DESTINATION = 1U << 13U
    };
    enum
    {
        RACHIS_LOAD_OP_LOAD = 0,
        RACHIS_LOAD_OP_CLEAR = 1,
        RACHIS_LOAD_OP_DISCARD = 2
    };
    enum
    {
        RACHIS_STORE_OP_STORE = 0,
        RACHIS_STORE_OP_DISCARD = 1
    };
    enum
    {
        RACHIS_SHADER_VERTEX = 1U << 0U,
        RACHIS_SHADER_PIXEL = 1U << 1U,
        RACHIS_SHADER_COMPUTE = 1U << 2U,
        RACHIS_SHADER_ALL = 0x7U
    };
    enum
    {
        RACHIS_BINDING_UNIFORM_BUFFER = 0,
        RACHIS_BINDING_READ_ONLY_STORAGE_BUFFER = 1,
        RACHIS_BINDING_STORAGE_BUFFER = 2,
        RACHIS_BINDING_SAMPLED_TEXTURE = 3,
        RACHIS_BINDING_STORAGE_TEXTURE = 4,
        RACHIS_BINDING_SAMPLER = 5
    };
    enum
    {
        RACHIS_FILTER_NEAREST = 0,
        RACHIS_FILTER_LINEAR = 1
    };
    enum
    {
        RACHIS_MIP_FILTER_NONE = 0,
        RACHIS_MIP_FILTER_NEAREST = 1,
        RACHIS_MIP_FILTER_LINEAR = 2
    };
    enum
    {
        RACHIS_ADDRESS_REPEAT = 0,
        RACHIS_ADDRESS_MIRROR_REPEAT = 1,
        RACHIS_ADDRESS_CLAMP_TO_EDGE = 2,
        RACHIS_ADDRESS_CLAMP_TO_BORDER = 3
    };
    enum
    {
        RACHIS_TOPOLOGY_POINT_LIST = 0,
        RACHIS_TOPOLOGY_LINE_LIST = 1,
        RACHIS_TOPOLOGY_LINE_STRIP = 2,
        RACHIS_TOPOLOGY_TRIANGLE_LIST = 3,
        RACHIS_TOPOLOGY_TRIANGLE_STRIP = 4
    };
    enum
    {
        RACHIS_CULL_NONE = 0,
        RACHIS_CULL_FRONT = 1,
        RACHIS_CULL_BACK = 2
    };
    enum
    {
        RACHIS_COMPARE_NEVER = 0,
        RACHIS_COMPARE_LESS = 1,
        RACHIS_COMPARE_EQUAL = 2,
        RACHIS_COMPARE_LESS_EQUAL = 3,
        RACHIS_COMPARE_GREATER = 4,
        RACHIS_COMPARE_NOT_EQUAL = 5,
        RACHIS_COMPARE_GREATER_EQUAL = 6,
        RACHIS_COMPARE_ALWAYS = 7
    };
    enum
    {
        RACHIS_BLEND_ZERO = 0,
        RACHIS_BLEND_ONE = 1,
        RACHIS_BLEND_SOURCE = 2,
        RACHIS_BLEND_ONE_MINUS_SOURCE = 3,
        RACHIS_BLEND_SOURCE_ALPHA = 4,
        RACHIS_BLEND_ONE_MINUS_SOURCE_ALPHA = 5,
        RACHIS_BLEND_DESTINATION = 6,
        RACHIS_BLEND_ONE_MINUS_DESTINATION = 7,
        RACHIS_BLEND_DESTINATION_ALPHA = 8,
        RACHIS_BLEND_ONE_MINUS_DESTINATION_ALPHA = 9,
        RACHIS_BLEND_SOURCE_ALPHA_SATURATED = 10,
        RACHIS_BLEND_CONSTANT = 11,
        RACHIS_BLEND_ONE_MINUS_CONSTANT = 12,
        RACHIS_BLEND_SOURCE1 = 13,
        RACHIS_BLEND_ONE_MINUS_SOURCE1 = 14,
        RACHIS_BLEND_SOURCE1_ALPHA = 15,
        RACHIS_BLEND_ONE_MINUS_SOURCE1_ALPHA = 16
    };
    enum
    {
        RACHIS_BLEND_OP_ADD = 0,
        RACHIS_BLEND_OP_SUBTRACT = 1,
        RACHIS_BLEND_OP_REVERSE_SUBTRACT = 2,
        RACHIS_BLEND_OP_MINIMUM = 3,
        RACHIS_BLEND_OP_MAXIMUM = 4
    };
    enum
    {
        RACHIS_COLOR_WRITE_RED = 1U << 0U,
        RACHIS_COLOR_WRITE_GREEN = 1U << 1U,
        RACHIS_COLOR_WRITE_BLUE = 1U << 2U,
        RACHIS_COLOR_WRITE_ALPHA = 1U << 3U,
        RACHIS_COLOR_WRITE_ALL = 0xfU
    };
    enum
    {
        RACHIS_INDEX_FORMAT_UINT16 = 0,
        RACHIS_INDEX_FORMAT_UINT32 = 1
    };
    enum
    {
        RACHIS_STRIP_INDEX_NONE = 0,
        RACHIS_STRIP_INDEX_UINT16 = 1,
        RACHIS_STRIP_INDEX_UINT32 = 2
    };
    enum
    {
        RACHIS_VERTEX_STEP_VERTEX = 0,
        RACHIS_VERTEX_STEP_INSTANCE = 1
    };
    enum
    {
        RACHIS_VERTEX_FORMAT_FLOAT32 = 0,
        RACHIS_VERTEX_FORMAT_FLOAT32X2 = 1,
        RACHIS_VERTEX_FORMAT_FLOAT32X3 = 2,
        RACHIS_VERTEX_FORMAT_FLOAT32X4 = 3,
        RACHIS_VERTEX_FORMAT_UINT32 = 4,
        RACHIS_VERTEX_FORMAT_UINT32X2 = 5,
        RACHIS_VERTEX_FORMAT_UINT32X3 = 6,
        RACHIS_VERTEX_FORMAT_UINT32X4 = 7,
        RACHIS_VERTEX_FORMAT_SINT32 = 8,
        RACHIS_VERTEX_FORMAT_SINT32X2 = 9,
        RACHIS_VERTEX_FORMAT_SINT32X3 = 10,
        RACHIS_VERTEX_FORMAT_SINT32X4 = 11,
        RACHIS_VERTEX_FORMAT_UINT8X2 = 12,
        RACHIS_VERTEX_FORMAT_UINT8X4 = 13,
        RACHIS_VERTEX_FORMAT_SINT8X2 = 14,
        RACHIS_VERTEX_FORMAT_SINT8X4 = 15,
        RACHIS_VERTEX_FORMAT_UNORM8X2 = 16,
        RACHIS_VERTEX_FORMAT_UNORM8X4 = 17,
        RACHIS_VERTEX_FORMAT_SNORM8X2 = 18,
        RACHIS_VERTEX_FORMAT_SNORM8X4 = 19,
        RACHIS_VERTEX_FORMAT_UINT16X2 = 20,
        RACHIS_VERTEX_FORMAT_UINT16X4 = 21,
        RACHIS_VERTEX_FORMAT_SINT16X2 = 22,
        RACHIS_VERTEX_FORMAT_SINT16X4 = 23,
        RACHIS_VERTEX_FORMAT_UNORM16X2 = 24,
        RACHIS_VERTEX_FORMAT_UNORM16X4 = 25,
        RACHIS_VERTEX_FORMAT_SNORM16X2 = 26,
        RACHIS_VERTEX_FORMAT_SNORM16X4 = 27,
        RACHIS_VERTEX_FORMAT_FLOAT16X2 = 28,
        RACHIS_VERTEX_FORMAT_FLOAT16X4 = 29
    };
    enum
    {
        RACHIS_DEVICE_INVALID = 0,
        RACHIS_DEVICE_ACTIVE = 1,
        RACHIS_DEVICE_LOST = 2
    };

    typedef struct rachis_error
    {
        uint32_t struct_size;
        rachis_status code;
        int64_t native_code;
        char message[RACHIS_ERROR_MESSAGE_CAPACITY];
    } rachis_error;

    typedef void (*rachis_diagnostic_callback)(uint32_t severity,
                                               uint32_t native_id,
                                               const char* message,
                                               void* user_data);

    typedef struct rachis_device_desc
    {
        uint32_t struct_size;
        uint32_t abi_version;
        uint32_t backend;
        uint32_t adapter_preference;
        uint32_t validation_level;
        uint32_t require_validation;
        uint32_t enable_dred;
        uint32_t enable_device_fault;
        uint64_t adapter_id;
        const char* application_name;
        const char* engine_name;
        rachis_diagnostic_callback diagnostic_callback;
        void* diagnostic_user_data;
    } rachis_device_desc;

    typedef struct rachis_device_capabilities
    {
        uint32_t struct_size;
        uint32_t unified_memory;
        uint32_t persistent_mapping;
        uint32_t mapped_memory_host_coherent;
        uint64_t mapped_memory_non_coherent_atom_size;
        uint64_t vertex_formats;
        uint32_t max_vertex_buffers;
        uint32_t max_vertex_attributes;
        uint32_t max_vertex_buffer_stride;
        uint32_t max_vertex_attribute_offset;
        uint32_t max_vertex_instance_step_rate;
        uint32_t reserved0;
    } rachis_device_capabilities;

    typedef struct rachis_buffer_desc
    {
        uint32_t struct_size;
        uint32_t abi_version;
        uint64_t size;
        uint32_t usage;
        uint32_t memory_type;
        const char* label;
    } rachis_buffer_desc;

    typedef struct rachis_mapped_memory
    {
        uint32_t struct_size;
        void* data;
        uint64_t size;
    } rachis_mapped_memory;

    typedef struct rachis_mapped_range
    {
        uint64_t offset;
        uint64_t size;
    } rachis_mapped_range;

    typedef struct rachis_texture_desc
    {
        uint32_t struct_size;
        uint32_t abi_version;
        uint32_t dimension;
        uint32_t format;
        uint32_t usage;
        uint32_t width;
        uint32_t height;
        uint32_t depth_or_array_layers;
        uint32_t mip_levels;
        uint32_t sample_count;
        const char* label;
    } rachis_texture_desc;

    typedef struct rachis_texture_view_desc
    {
        uint32_t struct_size;
        uint32_t abi_version;
        rachis_handle texture;
        uint32_t usage;
        uint32_t dimension;
        uint32_t format;
        uint32_t aspect;
        uint32_t first_mip_level;
        uint32_t mip_level_count;
        uint32_t first_array_layer;
        uint32_t array_layer_count;
        const char* label;
    } rachis_texture_view_desc;

    typedef struct rachis_command_pool_desc
    {
        uint32_t struct_size;
        uint32_t abi_version;
        uint32_t queue_type;
        uint32_t max_command_buffers;
        const char* label;
    } rachis_command_pool_desc;

    typedef struct rachis_surface_desc
    {
        uint32_t struct_size;
        uint32_t abi_version;
        uint32_t kind;
        void* window_handle;
        void* display_handle;
        const char* label;
    } rachis_surface_desc;

    typedef struct rachis_surface_format
    {
        uint32_t format;
        uint32_t color_space;
    } rachis_surface_format;

    typedef struct rachis_surface_capabilities
    {
        uint32_t struct_size;
        uint32_t min_image_count;
        uint32_t max_image_count;
        uint32_t min_width;
        uint32_t min_height;
        uint32_t max_width;
        uint32_t max_height;
        uint32_t preferred_width;
        uint32_t preferred_height;
        uint32_t usages;
        uint32_t present_modes;
        uint32_t format_count;
        rachis_surface_format formats[RACHIS_MAX_SURFACE_FORMATS];
    } rachis_surface_capabilities;

    typedef struct rachis_swapchain_desc
    {
        uint32_t struct_size;
        uint32_t abi_version;
        rachis_handle surface;
        uint32_t width;
        uint32_t height;
        uint32_t image_count;
        uint32_t format;
        uint32_t color_space;
        uint32_t usage;
        uint32_t present_mode;
        const char* label;
    } rachis_swapchain_desc;

    typedef struct rachis_swapchain_image
    {
        uint32_t struct_size;
        rachis_handle texture;
        rachis_handle render_target_view;
        uint32_t image_index;
        uint32_t width;
        uint32_t height;
        uint32_t format;
        uint32_t color_space;
        uint32_t initial_layout;
        uint32_t initial_stages;
        uint32_t initial_access;
        uint32_t required_final_layout;
        uint32_t required_final_stages;
        uint32_t required_final_access;
        uint32_t contents_defined;
        uint64_t generation;
    } rachis_swapchain_image;

    typedef struct rachis_diagnostic_counts
    {
        uint32_t struct_size;
        uint64_t information;
        uint64_t warnings;
        uint64_t errors;
        uint64_t corruptions;
    } rachis_diagnostic_counts;

    typedef struct rachis_adapter_info
    {
        uint32_t struct_size;
        uint32_t backend;
        uint64_t id;
        uint32_t vendor_id;
        uint32_t device_id;
        uint32_t type;
        uint32_t reserved0;
        uint64_t dedicated_video_memory;
        uint64_t shared_system_memory;
        char name[256];
    } rachis_adapter_info;

    typedef struct rachis_global_barrier
    {
        uint32_t stages_before;
        uint32_t stages_after;
        uint32_t access_before;
        uint32_t access_after;
    } rachis_global_barrier;

    typedef struct rachis_buffer_barrier
    {
        rachis_handle buffer;
        uint32_t stages_before;
        uint32_t stages_after;
        uint32_t access_before;
        uint32_t access_after;
        uint64_t offset;
        uint64_t size;
    } rachis_buffer_barrier;

    typedef struct rachis_texture_barrier
    {
        rachis_handle texture;
        uint32_t stages_before;
        uint32_t stages_after;
        uint32_t access_before;
        uint32_t access_after;
        uint32_t layout_before;
        uint32_t layout_after;
        uint32_t aspect;
        uint32_t first_mip_level;
        uint32_t mip_level_count;
        uint32_t first_array_layer;
        uint32_t array_layer_count;
    } rachis_texture_barrier;

    typedef struct rachis_barrier_batch
    {
        uint32_t struct_size;
        uint32_t abi_version;
        const rachis_global_barrier* global_barriers;
        uint32_t global_barrier_count;
        const rachis_buffer_barrier* buffer_barriers;
        uint32_t buffer_barrier_count;
        const rachis_texture_barrier* texture_barriers;
        uint32_t texture_barrier_count;
    } rachis_barrier_batch;

    typedef struct rachis_color_attachment
    {
        rachis_handle view;
        uint32_t load_op;
        uint32_t store_op;
        float clear_color[4];
    } rachis_color_attachment;

    typedef struct rachis_depth_stencil_attachment
    {
        rachis_handle view;
        uint32_t depth_load_op;
        uint32_t depth_store_op;
        float depth_clear_value;
        uint32_t stencil_load_op;
        uint32_t stencil_store_op;
        uint32_t stencil_clear_value;
    } rachis_depth_stencil_attachment;

    typedef struct rachis_rendering_desc
    {
        uint32_t struct_size;
        uint32_t abi_version;
        const rachis_color_attachment* color_attachments;
        uint32_t color_attachment_count;
        const rachis_depth_stencil_attachment* depth_stencil_attachment;
    } rachis_rendering_desc;

    typedef struct rachis_buffer_texture_region
    {
        uint64_t buffer_offset;
        uint32_t bytes_per_row;
        uint32_t rows_per_image;
        uint32_t mip_level;
        uint32_t array_layer;
        uint32_t aspect;
        uint32_t x;
        uint32_t y;
        uint32_t z;
        uint32_t width;
        uint32_t height;
        uint32_t depth;
    } rachis_buffer_texture_region;

    typedef struct rachis_buffer_texture_copy_desc
    {
        uint32_t struct_size;
        uint32_t abi_version;
        rachis_handle buffer;
        rachis_handle texture;
        const rachis_buffer_texture_region* regions;
        uint32_t region_count;
    } rachis_buffer_texture_copy_desc;

    typedef struct rachis_texture_transfer_layout
    {
        uint32_t struct_size;
        uint32_t bytes_per_row;
        uint32_t rows_per_image;
        uint32_t row_count;
        uint64_t row_data_size;
        uint64_t bytes_per_image;
        uint64_t required_buffer_size;
        uint32_t buffer_offset_alignment;
        uint32_t row_pitch_alignment;
    } rachis_texture_transfer_layout;

    typedef struct rachis_binding_layout_entry
    {
        uint32_t binding;
        uint32_t type;
        uint32_t visibility;
        uint32_t count;
    } rachis_binding_layout_entry;

    typedef struct rachis_binding_layout_desc
    {
        uint32_t struct_size;
        uint32_t abi_version;
        const rachis_binding_layout_entry* entries;
        uint32_t entry_count;
        const char* label;
    } rachis_binding_layout_desc;

    typedef struct rachis_binding_pool_size
    {
        uint32_t type;
        uint32_t descriptor_count;
    } rachis_binding_pool_size;

    typedef struct rachis_binding_pool_desc
    {
        uint32_t struct_size;
        uint32_t abi_version;
        uint32_t max_binding_sets;
        const rachis_binding_pool_size* sizes;
        uint32_t size_count;
        const char* label;
    } rachis_binding_pool_desc;

    typedef struct rachis_buffer_binding
    {
        uint32_t binding;
        uint32_t array_element;
        rachis_handle buffer;
        uint64_t offset;
        uint64_t size;
        uint32_t structure_byte_stride;
    } rachis_buffer_binding;

    typedef struct rachis_texture_binding
    {
        uint32_t binding;
        uint32_t array_element;
        rachis_handle view;
    } rachis_texture_binding;

    typedef struct rachis_sampler_binding
    {
        uint32_t binding;
        uint32_t array_element;
        rachis_handle sampler;
    } rachis_sampler_binding;

    typedef struct rachis_sampler_desc
    {
        uint32_t struct_size;
        uint32_t abi_version;
        uint32_t min_filter;
        uint32_t mag_filter;
        uint32_t mip_filter;
        uint32_t address_u;
        uint32_t address_v;
        uint32_t address_w;
        const char* label;
    } rachis_sampler_desc;

    typedef struct rachis_binding_set_desc
    {
        uint32_t struct_size;
        uint32_t abi_version;
        rachis_handle layout;
        rachis_handle pool;
        const rachis_buffer_binding* writes;
        uint32_t write_count;
        const char* label;
        const rachis_texture_binding* texture_writes;
        uint32_t texture_write_count;
        const rachis_sampler_binding* sampler_writes;
        uint32_t sampler_write_count;
    } rachis_binding_set_desc;

    typedef struct rachis_shader_binding_requirement
    {
        uint32_t set;
        uint32_t binding;
        uint32_t type;
        uint32_t count;
    } rachis_shader_binding_requirement;

    typedef struct rachis_shader_push_constant_requirement
    {
        uint32_t offset;
        uint32_t size;
    } rachis_shader_push_constant_requirement;

    typedef struct rachis_shader_module_desc
    {
        uint32_t struct_size;
        uint32_t abi_version;
        uint32_t stage;
        const void* bytecode;
        uint64_t bytecode_size;
        const char* entry_point;
        const rachis_shader_binding_requirement* bindings;
        uint32_t binding_count;
        uint32_t workgroup_x;
        uint32_t workgroup_y;
        uint32_t workgroup_z;
        const char* label;
        const rachis_shader_push_constant_requirement* push_constants;
        uint32_t push_constant_count;
        const uint32_t* vertex_input_locations;
        uint32_t vertex_input_location_count;
    } rachis_shader_module_desc;

    typedef struct rachis_push_constant_range
    {
        uint32_t visibility;
        uint32_t offset;
        uint32_t size;
    } rachis_push_constant_range;

    typedef struct rachis_pipeline_layout_desc
    {
        uint32_t struct_size;
        uint32_t abi_version;
        const rachis_handle* binding_layouts;
        uint32_t binding_layout_count;
        const char* label;
        const rachis_push_constant_range* push_constant_ranges;
        uint32_t push_constant_range_count;
    } rachis_pipeline_layout_desc;

    typedef struct rachis_compute_pipeline_desc
    {
        uint32_t struct_size;
        uint32_t abi_version;
        rachis_handle layout;
        rachis_handle shader;
        const char* label;
    } rachis_compute_pipeline_desc;

    typedef struct rachis_blend_component
    {
        uint32_t source_factor;
        uint32_t destination_factor;
        uint32_t operation;
    } rachis_blend_component;

    typedef struct rachis_color_target
    {
        uint32_t format;
        uint32_t write_mask;
        uint32_t blend_enabled;
        rachis_blend_component color_blend;
        rachis_blend_component alpha_blend;
    } rachis_color_target;

    typedef struct rachis_depth_stencil_state
    {
        uint32_t format;
        uint32_t depth_test_enabled;
        uint32_t depth_write_enabled;
        uint32_t depth_compare;
    } rachis_depth_stencil_state;

    typedef struct rachis_vertex_attribute
    {
        uint32_t format;
        uint32_t offset;
        uint32_t shader_location;
    } rachis_vertex_attribute;

    typedef struct rachis_vertex_buffer_layout
    {
        uint32_t stride;
        uint32_t step_mode;
        uint32_t instance_step_rate;
        const rachis_vertex_attribute* attributes;
        uint32_t attribute_count;
    } rachis_vertex_buffer_layout;

    typedef struct rachis_vertex_buffer_binding
    {
        rachis_handle buffer;
        uint64_t offset;
    } rachis_vertex_buffer_binding;

    typedef struct rachis_graphics_pipeline_desc
    {
        uint32_t struct_size;
        uint32_t abi_version;
        rachis_handle layout;
        rachis_handle vertex_shader;
        rachis_handle pixel_shader;
        uint32_t topology;
        uint32_t cull_mode;
        const rachis_color_target* color_targets;
        uint32_t color_target_count;
        const char* label;
        const rachis_depth_stencil_state* depth_stencil;
        const rachis_vertex_buffer_layout* vertex_buffers;
        uint32_t vertex_buffer_count;
        uint32_t strip_index_format;
    } rachis_graphics_pipeline_desc;

    typedef struct rachis_viewport
    {
        float x;
        float y;
        float width;
        float height;
        float min_depth;
        float max_depth;
    } rachis_viewport;

    typedef struct rachis_scissor
    {
        int32_t x;
        int32_t y;
        uint32_t width;
        uint32_t height;
    } rachis_scissor;

    typedef struct rachis_timestamp_pool_desc
    {
        uint32_t struct_size;
        uint32_t abi_version;
        uint32_t queue_type;
        uint32_t query_count;
        const char* label;
    } rachis_timestamp_pool_desc;

    RACHIS_API uint32_t rachis_abi_version(void);
    RACHIS_API rachis_status rachis_device_create(const rachis_device_desc* desc,
                                                  rachis_device** output,
                                                  rachis_error* error);

    RACHIS_API void rachis_device_destroy(rachis_device* device);
    RACHIS_API rachis_status rachis_device_diagnostic_counts(rachis_device* device,
                                                             rachis_diagnostic_counts* output,
                                                             rachis_error* error);
    RACHIS_API rachis_status rachis_device_operational_state(rachis_device* device,
                                                             uint32_t* output,
                                                             rachis_error* error);

    RACHIS_API rachis_status rachis_device_backend(rachis_device* device, uint32_t* output, rachis_error* error);
    RACHIS_API rachis_status rachis_device_adapter_info(rachis_device* device,
                                                        rachis_adapter_info* output,
                                                        rachis_error* error);
    RACHIS_API rachis_status rachis_device_loss_error(rachis_device* device,
                                                      rachis_error* loss_error,
                                                      rachis_error* error);
    RACHIS_API rachis_status rachis_device_query_capabilities(rachis_device* device,
                                                              rachis_device_capabilities* output,
                                                              rachis_error* error);

    RACHIS_API rachis_status rachis_queue_timestamp_frequency(rachis_device* device,
                                                              uint32_t queue_type,
                                                              uint64_t* output,
                                                              rachis_error* error);

    RACHIS_API rachis_status rachis_buffer_create(rachis_device* device,
                                                  const rachis_buffer_desc* desc,
                                                  rachis_handle* output,
                                                  rachis_error* error);
    RACHIS_API rachis_status rachis_buffer_destroy(rachis_device* device, rachis_handle buffer, rachis_error* error);

    RACHIS_API rachis_status rachis_buffer_write(rachis_device* device,
                                                 rachis_handle buffer,
                                                 uint64_t offset,
                                                 const void* bytes,
                                                 uint64_t size,
                                                 rachis_error* error);
    RACHIS_API rachis_status rachis_buffer_read(
        rachis_device* device, rachis_handle buffer, uint64_t offset, void* bytes, uint64_t size, rachis_error* error);
    RACHIS_API rachis_status rachis_buffer_map(rachis_device* device,
                                               rachis_handle buffer,
                                               uint64_t offset,
                                               uint64_t size,
                                               rachis_mapped_memory* output,
                                               rachis_error* error);
    RACHIS_API rachis_status rachis_buffer_unmap(rachis_device* device, rachis_handle buffer, rachis_error* error);
    RACHIS_API rachis_status rachis_buffer_flush_mapped_ranges(rachis_device* device,
                                                               rachis_handle buffer,
                                                               const rachis_mapped_range* ranges,
                                                               uint32_t range_count,
                                                               rachis_error* error);
    RACHIS_API rachis_status rachis_buffer_invalidate_mapped_ranges(rachis_device* device,
                                                                    rachis_handle buffer,
                                                                    const rachis_mapped_range* ranges,
                                                                    uint32_t range_count,
                                                                    rachis_error* error);

    RACHIS_API rachis_status rachis_texture_create(rachis_device* device,
                                                   const rachis_texture_desc* desc,
                                                   rachis_handle* output,
                                                   rachis_error* error);
    RACHIS_API rachis_status rachis_texture_destroy(rachis_device* device, rachis_handle texture, rachis_error* error);

    RACHIS_API rachis_status rachis_texture_lease_acquire(rachis_device* device,
                                                          rachis_handle texture,
                                                          rachis_handle* lease,
                                                          rachis_error* error);
    RACHIS_API rachis_status rachis_texture_lease_release(rachis_device* device,
                                                          rachis_handle lease,
                                                          rachis_error* error);
    RACHIS_API rachis_status rachis_texture_view_create(rachis_device* device,
                                                        const rachis_texture_view_desc* desc,
                                                        rachis_handle* output,
                                                        rachis_error* error);
    RACHIS_API rachis_status rachis_texture_view_destroy(rachis_device* device,
                                                         rachis_handle view,
                                                         rachis_error* error);
    RACHIS_API rachis_status rachis_texture_query_transfer_layout(rachis_device* device,
                                                                  rachis_handle texture,
                                                                  const rachis_buffer_texture_region* region,
                                                                  rachis_texture_transfer_layout* output,
                                                                  rachis_error* error);

    RACHIS_API rachis_status rachis_command_pool_create(rachis_device* device,
                                                        const rachis_command_pool_desc* desc,
                                                        rachis_handle* output,
                                                        rachis_error* error);
    RACHIS_API rachis_status rachis_command_pool_destroy(rachis_device* device,
                                                         rachis_handle pool,
                                                         rachis_error* error);
    RACHIS_API rachis_status rachis_command_acquire(rachis_device* device,
                                                    rachis_handle pool,
                                                    rachis_handle* output,
                                                    rachis_error* error);
    RACHIS_API rachis_status rachis_command_copy_buffer(rachis_device* device,
                                                        rachis_handle command,
                                                        rachis_handle source,
                                                        rachis_handle destination,
                                                        uint64_t source_offset,
                                                        uint64_t destination_offset,
                                                        uint64_t size,
                                                        rachis_error* error);
    RACHIS_API rachis_status rachis_command_copy_buffer_to_texture(rachis_device* device,
                                                                   rachis_handle command,
                                                                   const rachis_buffer_texture_copy_desc* desc,
                                                                   rachis_error* error);
    RACHIS_API rachis_status rachis_command_copy_texture_to_buffer(rachis_device* device,
                                                                   rachis_handle command,
                                                                   const rachis_buffer_texture_copy_desc* desc,
                                                                   rachis_error* error);

    RACHIS_API rachis_status rachis_command_barriers(rachis_device* device,
                                                     rachis_handle command,
                                                     const rachis_barrier_batch* batch,
                                                     rachis_error* error);

    RACHIS_API rachis_status rachis_command_begin_rendering(rachis_device* device,
                                                            rachis_handle command,
                                                            const rachis_rendering_desc* desc,
                                                            rachis_error* error);
    RACHIS_API rachis_status rachis_command_end_rendering(rachis_device* device,
                                                          rachis_handle command,
                                                          rachis_error* error);
    RACHIS_API rachis_status rachis_command_discard(rachis_device* device, rachis_handle command, rachis_error* error);
    RACHIS_API rachis_status rachis_command_submit(rachis_device* device,
                                                   rachis_handle command,
                                                   rachis_handle* timeline,
                                                   rachis_error* error);
    RACHIS_API rachis_status rachis_timeline_wait(rachis_device* device, rachis_handle timeline, rachis_error* error);
    RACHIS_API rachis_status rachis_timeline_release(rachis_device* device,
                                                     rachis_handle timeline,
                                                     rachis_error* error);

    RACHIS_API rachis_status rachis_binding_layout_create(rachis_device* device,
                                                          const rachis_binding_layout_desc* desc,
                                                          rachis_handle* output,
                                                          rachis_error* error);
    RACHIS_API rachis_status rachis_binding_layout_destroy(rachis_device* device,
                                                           rachis_handle layout,
                                                           rachis_error* error);
    RACHIS_API rachis_status rachis_binding_pool_create(rachis_device* device,
                                                        const rachis_binding_pool_desc* desc,
                                                        rachis_handle* output,
                                                        rachis_error* error);
    RACHIS_API rachis_status rachis_binding_pool_destroy(rachis_device* device,
                                                         rachis_handle pool,
                                                         rachis_error* error);
    RACHIS_API rachis_status rachis_binding_set_create(rachis_device* device,
                                                       const rachis_binding_set_desc* desc,
                                                       rachis_handle* output,
                                                       rachis_error* error);
    RACHIS_API rachis_status rachis_binding_set_destroy(rachis_device* device, rachis_handle set, rachis_error* error);
    RACHIS_API rachis_status rachis_sampler_create(rachis_device* device,
                                                   const rachis_sampler_desc* desc,
                                                   rachis_handle* output,
                                                   rachis_error* error);
    RACHIS_API rachis_status rachis_sampler_destroy(rachis_device* device, rachis_handle sampler, rachis_error* error);
    RACHIS_API rachis_status rachis_shader_module_create(rachis_device* device,
                                                         const rachis_shader_module_desc* desc,
                                                         rachis_handle* output,
                                                         rachis_error* error);
    RACHIS_API rachis_status rachis_shader_module_destroy(rachis_device* device,
                                                          rachis_handle shader,
                                                          rachis_error* error);
    RACHIS_API rachis_status rachis_pipeline_layout_create(rachis_device* device,
                                                           const rachis_pipeline_layout_desc* desc,
                                                           rachis_handle* output,
                                                           rachis_error* error);
    RACHIS_API rachis_status rachis_pipeline_layout_destroy(rachis_device* device,
                                                            rachis_handle layout,
                                                            rachis_error* error);
    RACHIS_API rachis_status rachis_compute_pipeline_create(rachis_device* device,
                                                            const rachis_compute_pipeline_desc* desc,
                                                            rachis_handle* output,
                                                            rachis_error* error);
    RACHIS_API rachis_status rachis_graphics_pipeline_create(rachis_device* device,
                                                             const rachis_graphics_pipeline_desc* desc,
                                                             rachis_handle* output,
                                                             rachis_error* error);
    RACHIS_API rachis_status rachis_pipeline_destroy(rachis_device* device,
                                                     rachis_handle pipeline,
                                                     rachis_error* error);
    RACHIS_API rachis_status rachis_command_bind_compute_pipeline(rachis_device* device,
                                                                  rachis_handle command,
                                                                  rachis_handle pipeline,
                                                                  rachis_error* error);
    RACHIS_API rachis_status rachis_command_bind_graphics_pipeline(rachis_device* device,
                                                                   rachis_handle command,
                                                                   rachis_handle pipeline,
                                                                   rachis_error* error);
    RACHIS_API rachis_status rachis_command_bind_graphics_set(
        rachis_device* device, rachis_handle command, uint32_t set_index, rachis_handle set, rachis_error* error);
    RACHIS_API rachis_status rachis_command_set_graphics_push_constants(rachis_device* device,
                                                                        rachis_handle command,
                                                                        uint32_t visibility,
                                                                        uint32_t offset,
                                                                        const void* data,
                                                                        uint32_t size,
                                                                        rachis_error* error);
    RACHIS_API rachis_status rachis_command_set_viewport(rachis_device* device,
                                                         rachis_handle command,
                                                         const rachis_viewport* viewport,
                                                         rachis_error* error);
    RACHIS_API rachis_status rachis_command_set_scissor(rachis_device* device,
                                                        rachis_handle command,
                                                        const rachis_scissor* scissor,
                                                        rachis_error* error);
    RACHIS_API rachis_status rachis_command_draw(rachis_device* device,
                                                 rachis_handle command,
                                                 uint32_t vertex_count,
                                                 uint32_t instance_count,
                                                 uint32_t first_vertex,
                                                 uint32_t first_instance,
                                                 rachis_error* error);
    RACHIS_API rachis_status rachis_command_bind_vertex_buffers(rachis_device* device,
                                                                rachis_handle command,
                                                                uint32_t first_slot,
                                                                const rachis_vertex_buffer_binding* bindings,
                                                                uint32_t binding_count,
                                                                rachis_error* error);
    RACHIS_API rachis_status rachis_command_bind_index_buffer(rachis_device* device,
                                                              rachis_handle command,
                                                              rachis_handle buffer,
                                                              uint32_t index_format,
                                                              uint64_t offset,
                                                              rachis_error* error);
    RACHIS_API rachis_status rachis_command_draw_indexed(rachis_device* device,
                                                         rachis_handle command,
                                                         uint32_t index_count,
                                                         uint32_t instance_count,
                                                         uint32_t first_index,
                                                         int32_t base_vertex,
                                                         uint32_t first_instance,
                                                         rachis_error* error);
    RACHIS_API rachis_status rachis_command_bind_compute_set(
        rachis_device* device, rachis_handle command, uint32_t set_index, rachis_handle set, rachis_error* error);
    RACHIS_API rachis_status rachis_command_set_compute_push_constants(rachis_device* device,
                                                                       rachis_handle command,
                                                                       uint32_t offset,
                                                                       const void* data,
                                                                       uint32_t size,
                                                                       rachis_error* error);
    RACHIS_API rachis_status rachis_command_dispatch(
        rachis_device* device, rachis_handle command, uint32_t x, uint32_t y, uint32_t z, rachis_error* error);
    RACHIS_API rachis_status rachis_timestamp_pool_create(rachis_device* device,
                                                          const rachis_timestamp_pool_desc* desc,
                                                          rachis_handle* output,
                                                          rachis_error* error);
    RACHIS_API rachis_status rachis_query_pool_destroy(rachis_device* device, rachis_handle pool, rachis_error* error);

    RACHIS_API rachis_status rachis_command_reset_queries(rachis_device* device,
                                                          rachis_handle command,
                                                          rachis_handle pool,
                                                          uint32_t first_query,
                                                          uint32_t query_count,
                                                          rachis_error* error);
    RACHIS_API rachis_status rachis_command_write_timestamp(
        rachis_device* device, rachis_handle command, rachis_handle pool, uint32_t query_index, rachis_error* error);
    RACHIS_API rachis_status rachis_command_resolve_timestamps(rachis_device* device,
                                                               rachis_handle command,
                                                               rachis_handle pool,
                                                               uint32_t first_query,
                                                               uint32_t query_count,
                                                               rachis_handle destination,
                                                               uint64_t destination_offset,
                                                               rachis_error* error);

    RACHIS_API rachis_status rachis_surface_create(rachis_device* device,
                                                   const rachis_surface_desc* desc,
                                                   rachis_handle* output,
                                                   rachis_error* error);
    RACHIS_API rachis_status rachis_surface_query_capabilities(rachis_device* device,
                                                               rachis_handle surface,
                                                               rachis_surface_capabilities* output,
                                                               rachis_error* error);
    RACHIS_API rachis_status rachis_surface_destroy(rachis_device* device, rachis_handle surface, rachis_error* error);
    RACHIS_API rachis_status rachis_swapchain_create(rachis_device* device,
                                                     const rachis_swapchain_desc* desc,
                                                     rachis_handle* output,
                                                     rachis_error* error);
    RACHIS_API rachis_status rachis_swapchain_destroy(rachis_device* device,
                                                      rachis_handle swapchain,
                                                      rachis_error* error);
    RACHIS_API rachis_status rachis_swapchain_resize(
        rachis_device* device, rachis_handle swapchain, uint32_t width, uint32_t height, rachis_error* error);
    RACHIS_API rachis_status rachis_swapchain_acquire(rachis_device* device,
                                                      rachis_handle swapchain,
                                                      rachis_swapchain_image* output,
                                                      rachis_error* error);

    typedef struct rachis_swapchain_presentation_progress
    {
        uint32_t struct_size;
        uint32_t supported;
        uint64_t generation;
        uint64_t submitted_sequence;
        uint64_t displayed_sequence;
    } rachis_swapchain_presentation_progress;
    RACHIS_API rachis_status
    rachis_swapchain_query_presentation_progress(rachis_device* device,
                                                 rachis_handle swapchain,
                                                 rachis_swapchain_presentation_progress* output,
                                                 rachis_error* error);

    RACHIS_API rachis_status rachis_swapchain_cancel(rachis_device* device,
                                                     rachis_handle swapchain,
                                                     rachis_error* error);

    RACHIS_API rachis_status rachis_swapchain_present(rachis_device* device,
                                                      rachis_handle swapchain,
                                                      rachis_handle render_complete,
                                                      uint32_t* present_status,
                                                      rachis_error* error);

#ifdef __cplusplus
}
#endif

#endif
