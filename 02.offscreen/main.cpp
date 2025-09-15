/*
 * main.cpp — Vulkan offscreen quad → output.ppm (with sync + culling fixes)
 *
 * What this sample does
 * ---------------------
 * - Creates a minimal Vulkan instance/device with a graphics queue.
 * - Allocates an offscreen color VkImage as our render target.
 * - Builds a tiny graphics pipeline (vertex + fragment shader) that draws a quad from two triangles.
 * - Renders into the offscreen image via a single-subpass render pass.
 * - Copies the rendered RGBA8 pixels into a host-visible buffer.
 * - Writes a binary PPM (P6) file to disk (output.ppm), vertically flipped to the conventional top-left origin.
 *
 * Build & run (Debian/Ubuntu):
 *   sudo apt install build-essential libvulkan-dev vulkan-headers glslang-tools
 *   make
 *   ./hello_vulkan_offscreen
 *
 * Notes on correctness
 * --------------------
 * - Validation layers are enabled in debug builds. Warnings/errors are printed to stderr.
 * - Rasterizer culling is disabled (VK_CULL_MODE_NONE) so the quad is always visible regardless of vertex order.
 * - Explicit synchronization:
 *     • Before the render pass: transition image UNDEFINED → COLOR_ATTACHMENT_OPTIMAL.
 *     • After the render pass: ensure color writes are visible to transfer before copying (COLOR_ATTACHMENT_OUTPUT → TRANSFER).
 *     • Before host read: buffer barrier TRANSFER_WRITE → HOST_READ.
 * - The render pass finalLayout is TRANSFER_SRC_OPTIMAL to match the subsequent vkCmdCopyImageToBuffer.
 */

#include <vulkan/vulkan.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NDEBUG
static constexpr bool kEnableValidation = true;   // Enable validation in debug mode.
#else
static constexpr bool kEnableValidation = false;  // Disable validation in release.
#endif

// Helper macro: check VkResult and abort on failure so we don't silently continue in a bad state.
#define VK_CHECK(x) do { VkResult err = (x); if (err != VK_SUCCESS) { \
    std::cerr << "Vulkan error " << err << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::abort(); }} while (0)

static const char* kValidationLayer = "VK_LAYER_KHRONOS_validation"; // Standard Khronos validation layer name.

// Function pointers for EXT_debug_utils (loaded at runtime).
PFN_vkCreateDebugUtilsMessengerEXT  pfnCreateDebugUtilsMessengerEXT  = nullptr;
PFN_vkDestroyDebugUtilsMessengerEXT pfnDestroyDebugUtilsMessengerEXT = nullptr;

// Debug callback: print warnings/errors emitted by validation to stderr.
VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT /*types*/,
    const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
    void* /*userData*/) {
    if (severity & (VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)) {
        std::cerr << "[Vulkan] " << callbackData->pMessage << std::endl;
    }
    return VK_FALSE; // Never suppress messages.
}

// Utility: check whether a given instance layer is present.
bool HasLayer(const char* name) {
    uint32_t count = 0;
    VK_CHECK(vkEnumerateInstanceLayerProperties(&count, nullptr));
    std::vector<VkLayerProperties> props(count);
    VK_CHECK(vkEnumerateInstanceLayerProperties(&count, props.data()));
    for (auto& p : props) if (std::strcmp(p.layerName, name) == 0) return true;
    return false;
}

// Return the instance extensions we need. On Apple we also opt into portability.
std::vector<const char*> GetRequiredInstanceExtensions() {
    std::vector<const char*> exts;
    if (kEnableValidation) exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#ifdef __APPLE__
    exts.push_back("VK_KHR_portability_enumeration");
#endif
    return exts;
}

// Resolve the EXT_debug_utils entry points from the instance.
void LoadDebugUtils(VkInstance instance) {
    pfnCreateDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    pfnDestroyDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
}

// Create a debug messenger if validation is enabled.
VkDebugUtilsMessengerEXT CreateDebugMessenger(VkInstance instance) {
    if (!kEnableValidation || !pfnCreateDebugUtilsMessengerEXT) return VK_NULL_HANDLE;
    VkDebugUtilsMessengerCreateInfoEXT ci{};
    ci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    ci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    ci.pfnUserCallback = DebugCallback;
    VkDebugUtilsMessengerEXT m = VK_NULL_HANDLE;
    VK_CHECK(pfnCreateDebugUtilsMessengerEXT(instance, &ci, nullptr, &m));
    return m;
}

// --- Device selection helpers -------------------------------------------------

struct QueueFamilyIndices {
    std::optional<uint32_t> graphics;             // Index of a queue family that supports graphics commands.
    bool complete() const { return graphics.has_value(); }
};

// Pick any queue family that supports graphics.
QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice pd) {
    QueueFamilyIndices indices;
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, props.data());
    for (uint32_t i = 0; i < count; ++i) {
        if (props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { indices.graphics = i; break; }
    }
    return indices;
}

// Prefer a discrete GPU with a graphics queue; fall back to any GPU with graphics.
VkPhysicalDevice PickPhysicalDevice(VkInstance instance) {
    uint32_t count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, nullptr));
    if (count == 0) throw std::runtime_error("No Vulkan-capable GPUs found.");
    std::vector<VkPhysicalDevice> devices(count);
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, devices.data()));

    VkPhysicalDevice chosen = VK_NULL_HANDLE;
    for (auto d : devices) {
        VkPhysicalDeviceProperties props{}; vkGetPhysicalDeviceProperties(d, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && FindQueueFamilies(d).complete()) { chosen = d; break; }
    }
    if (chosen == VK_NULL_HANDLE) // If no discrete GPU, take the first with graphics support.
        for (auto d : devices) if (FindQueueFamilies(d).complete()) { chosen = d; break; }

    if (chosen == VK_NULL_HANDLE) throw std::runtime_error("No suitable GPU with graphics queue.");
    return chosen;
}

// Find a memory type index compatible with typeBits and properties.
uint32_t FindMemoryType(VkPhysicalDevice pd, uint32_t typeBits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(pd, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    throw std::runtime_error("No compatible memory type.");
}

// Read a whole file into memory (used to load SPIR-V shader binaries).
std::vector<char> ReadFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Failed to open file: " + path);
    size_t size = (size_t)f.tellg();
    std::vector<char> data(size);
    f.seekg(0);
    f.read(data.data(), size);
    return data;
}

int main() {
    const uint32_t WIDTH = 512, HEIGHT = 512; // Output resolution.
    try {
        // ---------------------------------------------------------------------
        // 1) Instance creation (plus optional validation setup)
        // ---------------------------------------------------------------------
        VkApplicationInfo appInfo{}; appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Vulkan Offscreen Quad"; appInfo.apiVersion = VK_API_VERSION_1_2;

        auto exts = GetRequiredInstanceExtensions();
        std::vector<const char*> layers;
        if (kEnableValidation && HasLayer(kValidationLayer)) layers.push_back(kValidationLayer);

        VkInstanceCreateInfo ici{}; ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &appInfo; 
        ici.enabledExtensionCount = (uint32_t)exts.size();
        ici.ppEnabledExtensionNames = exts.empty()? nullptr : exts.data();
        ici.enabledLayerCount = (uint32_t)layers.size(); 
        ici.ppEnabledLayerNames = layers.empty()? nullptr : layers.data();
#ifdef __APPLE__
        // On MoltenVK/Apple, portability enumeration must be enabled to see all devices.
        ici.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
        VkInstance instance{}; VK_CHECK(vkCreateInstance(&ici, nullptr, &instance));

        // Load EXT_debug_utils and start a messenger (no-op if validation disabled).
        LoadDebugUtils(instance); auto dbg = CreateDebugMessenger(instance);

        // ---------------------------------------------------------------------
        // 2) Physical device & logical device with one graphics queue
        // ---------------------------------------------------------------------
        VkPhysicalDevice phys = PickPhysicalDevice(instance);
        QueueFamilyIndices qfi = FindQueueFamilies(phys);
        uint32_t gfxQ = qfi.graphics.value();

        float prio = 1.0f; 
        VkDeviceQueueCreateInfo qci{}; qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = gfxQ; qci.queueCount = 1; qci.pQueuePriorities = &prio; // single queue is enough here.

        VkDeviceCreateInfo dci{}; dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO; 
        dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
#ifdef __APPLE__
        // MoltenVK portability subset requirement on some Apple setups.
        const char* devExts[] = {"VK_KHR_portability_subset"};
        dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = devExts;
#endif
        VkDevice device{}; VK_CHECK(vkCreateDevice(phys, &dci, nullptr, &device));
        VkQueue queue{}; vkGetDeviceQueue(device, gfxQ, 0, &queue);

        // ---------------------------------------------------------------------
        // 3) Command pool & command buffer (single-shot recording)
        // ---------------------------------------------------------------------
        VkCommandPoolCreateInfo poolCI{}; poolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO; poolCI.queueFamilyIndex = gfxQ;
        VkCommandPool pool{}; VK_CHECK(vkCreateCommandPool(device, &poolCI, nullptr, &pool));

        VkCommandBufferAllocateInfo cba{}; cba.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO; 
        cba.commandPool = pool; cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cba.commandBufferCount = 1;
        VkCommandBuffer cmd{}; VK_CHECK(vkAllocateCommandBuffers(device, &cba, &cmd));

        // ---------------------------------------------------------------------
        // 4) Offscreen color image (RGBA8) + view — acts as our render target
        // ---------------------------------------------------------------------
        VkImageCreateInfo imgCI{}; imgCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO; imgCI.imageType = VK_IMAGE_TYPE_2D;
        imgCI.extent = {WIDTH, HEIGHT, 1}; imgCI.mipLevels = 1; imgCI.arrayLayers = 1; imgCI.format = VK_FORMAT_R8G8B8A8_UNORM;
        imgCI.tiling = VK_IMAGE_TILING_OPTIMAL; imgCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imgCI.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT; // render to it, then read via transfer.
        imgCI.samples = VK_SAMPLE_COUNT_1_BIT; imgCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkImage colorImage{}; VK_CHECK(vkCreateImage(device, &imgCI, nullptr, &colorImage));

        // Allocate device-local memory for the image and bind it.
        VkMemoryRequirements imgMemReq{}; vkGetImageMemoryRequirements(device, colorImage, &imgMemReq);
        VkMemoryAllocateInfo imgAlloc{}; imgAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; imgAlloc.allocationSize = imgMemReq.size;
        imgAlloc.memoryTypeIndex = FindMemoryType(phys, imgMemReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceMemory colorMem{}; VK_CHECK(vkAllocateMemory(device, &imgAlloc, nullptr, &colorMem));
        VK_CHECK(vkBindImageMemory(device, colorImage, colorMem, 0));

        // Create an image view so the image can be used as a framebuffer attachment.
        VkImageViewCreateInfo ivCI{}; ivCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO; ivCI.image = colorImage;
        ivCI.viewType = VK_IMAGE_VIEW_TYPE_2D; ivCI.format = imgCI.format; ivCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkImageView colorView{}; VK_CHECK(vkCreateImageView(device, &ivCI, nullptr, &colorView));

        // ---------------------------------------------------------------------
        // 5) Render pass & framebuffer targeting the color image
        // ---------------------------------------------------------------------
        // Single color attachment we clear at load and store for later copying.
        VkAttachmentDescription colorAtt{}; colorAtt.format = imgCI.format; colorAtt.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; 
        colorAtt.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; // ready for vkCmdCopyImageToBuffer

        VkAttachmentReference colorRef{}; colorRef.attachment = 0; colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{}; subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS; 
        subpass.colorAttachmentCount = 1; subpass.pColorAttachments = &colorRef;

        // External → subpass dependency: makes color attachment writes happen in the right stage.
        VkSubpassDependency dep{}; dep.srcSubpass = VK_SUBPASS_EXTERNAL; dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT; 
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.srcAccessMask = 0; 
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo rpCI{}; rpCI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO; 
        rpCI.attachmentCount = 1; rpCI.pAttachments = &colorAtt; 
        rpCI.subpassCount = 1; rpCI.pSubpasses = &subpass; 
        rpCI.dependencyCount = 1; rpCI.pDependencies = &dep;
        VkRenderPass renderPass{}; VK_CHECK(vkCreateRenderPass(device, &rpCI, nullptr, &renderPass));

        // Framebuffer that binds the image view as the color attachment.
        VkFramebufferCreateInfo fbCI{}; fbCI.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO; fbCI.renderPass = renderPass;
        fbCI.attachmentCount = 1; fbCI.pAttachments = &colorView; fbCI.width = WIDTH; fbCI.height = HEIGHT; fbCI.layers = 1;
        VkFramebuffer fb{}; VK_CHECK(vkCreateFramebuffer(device, &fbCI, nullptr, &fb));

        // ---------------------------------------------------------------------
        // 6) Vertex/index buffers for a quad (two triangles in NDC)
        // ---------------------------------------------------------------------
        struct Vertex { float pos[2]; float col[3]; };
        const Vertex verts[] = {
            {{-0.5f, -0.5f}, {1.f, 0.f, 0.f}}, // bottom-left (red)
            {{ 0.5f, -0.5f}, {0.f, 1.f, 0.f}}, // bottom-right (green)
            {{ 0.5f,  0.5f}, {0.f, 0.f, 1.f}}, // top-right (blue)
            {{-0.5f,  0.5f}, {1.f, 1.f, 0.f}}, // top-left (yellow)
        };
        const uint16_t idx[] = {0,1,2, 2,3,0};

        // Small helper to create and allocate a buffer with given usage and memory props.
        auto createBuffer = [&](VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, VkBuffer& buf, VkDeviceMemory& mem){
            VkBufferCreateInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; bi.size = size; bi.usage = usage; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VK_CHECK(vkCreateBuffer(device, &bi, nullptr, &buf));
            VkMemoryRequirements mr{}; vkGetBufferMemoryRequirements(device, buf, &mr);
            VkMemoryAllocateInfo ai{}; ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO; ai.allocationSize = mr.size; ai.memoryTypeIndex = FindMemoryType(phys, mr.memoryTypeBits, props);
            VK_CHECK(vkAllocateMemory(device, &ai, nullptr, &mem));
            VK_CHECK(vkBindBufferMemory(device, buf, mem, 0));
        };

        // Host-visible/Coherent so we can map and fill data directly.
        VkBuffer vbo{}, ibo{}; VkDeviceMemory vmem{}, imem{};
        createBuffer(sizeof(verts), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, vbo, vmem);
        createBuffer(sizeof(idx),   VK_BUFFER_USAGE_INDEX_BUFFER_BIT,  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, ibo, imem);

        // Upload vertex/index data by mapping and memcpy.
        void* p=nullptr; vkMapMemory(device, vmem, 0, sizeof(verts), 0, &p); std::memcpy(p, verts, sizeof(verts)); vkUnmapMemory(device, vmem);
        vkMapMemory(device, imem, 0, sizeof(idx), 0, &p); std::memcpy(p, idx, sizeof(idx)); vkUnmapMemory(device, imem);

        // ---------------------------------------------------------------------
        // 7) Pipeline state objects (shader modules, fixed-function state)
        // ---------------------------------------------------------------------
        // Load precompiled SPIR-V shaders (Makefile should build quad.vert.spv / quad.frag.spv).
        auto vertCode = ReadFile("quad.vert.spv");
        auto fragCode = ReadFile("quad.frag.spv");
        auto makeShader = [&](const std::vector<char>& code){
            VkShaderModuleCreateInfo sm{}; sm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO; sm.codeSize = code.size(); sm.pCode = reinterpret_cast<const uint32_t*>(code.data());
            VkShaderModule mod{}; VK_CHECK(vkCreateShaderModule(device, &sm, nullptr, &mod)); return mod; };
        VkShaderModule vs = makeShader(vertCode);
        VkShaderModule fs = makeShader(fragCode);

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   stages[0].module = vs; stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = fs; stages[1].pName = "main";

        // Vertex layout: binding 0 with position (vec2) at location 0 and color (vec3) at location 1.
        VkVertexInputBindingDescription bind{}; bind.binding = 0; bind.stride = sizeof(Vertex); bind.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        VkVertexInputAttributeDescription attrs[2]{};
        attrs[0].location = 0; attrs[0].binding = 0; attrs[0].format = VK_FORMAT_R32G32_SFLOAT;    attrs[0].offset = offsetof(Vertex, pos);
        attrs[1].location = 1; attrs[1].binding = 0; attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT; attrs[1].offset = offsetof(Vertex, col);
        VkPipelineVertexInputStateCreateInfo vi{}; vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO; vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &bind; vi.vertexAttributeDescriptionCount = 2; vi.pVertexAttributeDescriptions = attrs;

        VkPipelineInputAssemblyStateCreateInfo ia{}; ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO; ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        // Fixed viewport/scissor to match our offscreen image extents.
        VkViewport vp{}; vp.x = 0; vp.y = 0; vp.width = static_cast<float>(WIDTH); vp.height = static_cast<float>(HEIGHT); vp.minDepth = 0.f; vp.maxDepth = 1.f;
        VkRect2D sc{}; sc.offset = {0,0}; sc.extent = {WIDTH, HEIGHT};
        VkPipelineViewportStateCreateInfo vpState{}; vpState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO; vpState.viewportCount = 1; vpState.pViewports = &vp; vpState.scissorCount = 1; vpState.pScissors = &sc;

        // Rasterization: culling disabled to avoid orientation issues; CCW front face (typical Vulkan NDC).
        VkPipelineRasterizationStateCreateInfo rs{}; rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO; rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;

        // No MSAA.
        VkPipelineMultisampleStateCreateInfo ms{}; ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO; ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        // Color blend: write RGBA, no blending.
        VkPipelineColorBlendAttachmentState cbAtt{}; cbAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT; cbAtt.blendEnable = VK_FALSE;
        VkPipelineColorBlendStateCreateInfo cb{}; cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO; cb.attachmentCount = 1; cb.pAttachments = &cbAtt;

        // No descriptor sets or push constants needed.
        VkPipelineLayoutCreateInfo plCI{}; plCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO; 
        VkPipelineLayout pipelineLayout{}; VK_CHECK(vkCreatePipelineLayout(device, &plCI, nullptr, &pipelineLayout));

        // No depth/stencil for a pure color pass.
        VkPipelineDepthStencilStateCreateInfo ds{}; ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO; ds.depthTestEnable = VK_FALSE; ds.depthWriteEnable = VK_FALSE;

        // Bake the graphics pipeline for our single-subpass render pass.
        VkGraphicsPipelineCreateInfo gp{}; gp.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO; gp.stageCount = 2; gp.pStages = stages; gp.pVertexInputState = &vi; gp.pInputAssemblyState = &ia; gp.pViewportState = &vpState; gp.pRasterizationState = &rs; gp.pMultisampleState = &ms; gp.pColorBlendState = &cb; gp.pDepthStencilState = &ds; gp.layout = pipelineLayout; gp.renderPass = renderPass; gp.subpass = 0;
        VkPipeline pipeline{}; VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gp, nullptr, &pipeline));

        // ---------------------------------------------------------------------
        // 8) Host-visible staging buffer for readback
        // ---------------------------------------------------------------------
        VkBuffer readback{}; VkDeviceMemory readMem{};
        VkDeviceSize pixelSize = 4; // RGBA8 → 4 bytes per pixel.
        VkDeviceSize bufSize = WIDTH * HEIGHT * pixelSize;
        createBuffer(bufSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, readback, readMem);

        // ---------------------------------------------------------------------
        // 9) Record commands: layout transitions → render → copy → host barrier
        // ---------------------------------------------------------------------
        VkCommandBufferBeginInfo begin{}; begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO; 
        VK_CHECK(vkBeginCommandBuffer(cmd, &begin));

        // (a) Transition color image: UNDEFINED → COLOR_ATTACHMENT_OPTIMAL before we render into it.
        VkImageMemoryBarrier toColor{}; toColor.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER; 
        toColor.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; 
        toColor.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; 
        toColor.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; 
        toColor.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; 
        toColor.image = colorImage; 
        toColor.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0,1,0,1};
        toColor.srcAccessMask = 0; 
        toColor.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, 
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,            // earliest possible
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,// we will write color
            0, 
            0,nullptr, 0,nullptr, 1,&toColor);

        // (b) Begin render pass and draw the indexed quad.
        VkClearValue clear{}; clear.color = {{0.25f, 0.25f, 0.3f, 1.0f}}; // Slightly dark bluish gray background.
        VkRenderPassBeginInfo rpBI{}; rpBI.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO; rpBI.renderPass = renderPass; rpBI.framebuffer = fb; rpBI.renderArea = {{0,0},{WIDTH,HEIGHT}}; rpBI.clearValueCount = 1; rpBI.pClearValues = &clear;
        vkCmdBeginRenderPass(cmd, &rpBI, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        VkDeviceSize off = 0; vkCmdBindVertexBuffers(cmd, 0, 1, &vbo, &off);
        vkCmdBindIndexBuffer(cmd, ibo, 0, VK_INDEX_TYPE_UINT16);
        vkCmdDrawIndexed(cmd, 6, 1, 0, 0, 0);
        vkCmdEndRenderPass(cmd);

        // (c) Ensure color writes are visible to transfer and that layout matches TRANSFER_SRC_OPTIMAL.
        //     Note: finalLayout already set to TRANSFER_SRC_OPTIMAL by the render pass; we reinforce visibility here.
        VkImageMemoryBarrier afterRP{}; afterRP.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER; 
        afterRP.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; // logically already in this layout after the subpass/end.
        afterRP.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; 
        afterRP.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; 
        afterRP.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; 
        afterRP.image = colorImage; 
        afterRP.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
        afterRP.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; 
        afterRP.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, 
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, // wait for color writes
            VK_PIPELINE_STAGE_TRANSFER_BIT,                // before transfer reads
            0, 0,nullptr, 0,nullptr, 1,&afterRP);

        // (d) Copy the image contents into the host-visible buffer.
        VkBufferImageCopy copy{}; copy.bufferOffset = 0; copy.bufferRowLength = 0; copy.bufferImageHeight = 0; // tightly packed
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; copy.imageOffset = {0,0,0}; copy.imageExtent = {WIDTH, HEIGHT, 1};
        vkCmdCopyImageToBuffer(cmd, colorImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1, &copy);

        // (e) Barrier to make TRANSFER_WRITEs visible to the host mapping.
        VkBufferMemoryBarrier toHost{}; toHost.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER; 
        toHost.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; 
        toHost.dstAccessMask = VK_ACCESS_HOST_READ_BIT; 
        toHost.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; 
        toHost.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; 
        toHost.buffer = readback; toHost.offset = 0; toHost.size = bufSize;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0,nullptr, 1,&toHost, 0,nullptr);

        VK_CHECK(vkEndCommandBuffer(cmd));

        // ---------------------------------------------------------------------
        // 10) Submit & wait for completion (simple fence)
        // ---------------------------------------------------------------------
        VkFenceCreateInfo fi{}; fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO; 
        VkFence fence{}; VK_CHECK(vkCreateFence(device, &fi, nullptr, &fence));
        VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        VK_CHECK(vkQueueSubmit(queue, 1, &si, fence)); 
        VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_C(5'000'000'000))); // 5s timeout

        // ---------------------------------------------------------------------
        // 11) Map the staging buffer and write PPM (P6). Flip vertically to top-left origin.
        // ---------------------------------------------------------------------
        void* mapped = nullptr; VK_CHECK(vkMapMemory(device, readMem, 0, bufSize, 0, &mapped));
        const uint8_t* src = reinterpret_cast<const uint8_t*>(mapped);
        std::ofstream ppm("output.ppm", std::ios::binary);
        ppm << "P6\n" << WIDTH << " " << HEIGHT << "\n255\n";
        for (int y = HEIGHT - 1; y >= 0; --y) { // flip vertically
            const uint8_t* row = src + y * WIDTH * 4; // 4 = RGBA8
            for (uint32_t x = 0; x < WIDTH; ++x) {
                ppm.put(row[x*4 + 0]); // R
                ppm.put(row[x*4 + 1]); // G
                ppm.put(row[x*4 + 2]); // B
            }
        }
        ppm.close();
        vkUnmapMemory(device, readMem);
        std::cout << "Wrote output.ppm (" << WIDTH << "x" << HEIGHT << ")" << std::endl;

        // ---------------------------------------------------------------------
        // 12) Cleanup (in reverse allocation order when possible)
        // ---------------------------------------------------------------------
        vkDestroyFence(device, fence, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyShaderModule(device, fs, nullptr);
        vkDestroyShaderModule(device, vs, nullptr);
        vkDestroyBuffer(device, readback, nullptr); vkFreeMemory(device, readMem, nullptr);
        vkDestroyBuffer(device, ibo, nullptr); vkFreeMemory(device, imem, nullptr);
        vkDestroyBuffer(device, vbo, nullptr); vkFreeMemory(device, vmem, nullptr);
        vkDestroyFramebuffer(device, fb, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);
        vkDestroyImageView(device, colorView, nullptr);
        vkDestroyImage(device, colorImage, nullptr);
        vkFreeMemory(device, colorMem, nullptr);
        vkDestroyCommandPool(device, pool, nullptr);
        vkDestroyDevice(device, nullptr);
        if (dbg && pfnDestroyDebugUtilsMessengerEXT) pfnDestroyDebugUtilsMessengerEXT(instance, dbg, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << std::endl; return 1;
    }
}