// main.cpp — Vulkan + GLFW: render a colored quad to a window (DETAILED COMMENTS)
// ---------------------------------------------------------------------------------
// This program creates a minimal Vulkan renderer using GLFW to open a window and
// display a colored quad. It covers:
//   • Instance / device creation (with optional validation)
//   • Surface and swapchain setup
//   • Graphics pipeline with vertex/index buffers
//   • Per-swapchain command buffers
//   • Frame synchronization (semaphores + fences)
//   • Resize handling with swapchain recreation
//
// Deps (Debian/Ubuntu):
//   sudo apt install build-essential libvulkan-dev vulkan-headers libglfw3-dev glslang-tools
//
// Shaders:
//   Compile GLSL → SPIR-V (example):
//     glslangValidator -V -o quad.vert.spv quad.vert
//     glslangValidator -V -o quad.frag.spv quad.frag
//
// Build: make
// Run:   ./vulkan_glfw_quad
// ---------------------------------------------------------------------------------

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NDEBUG
static constexpr bool kEnableValidation = true; // Enable debug layer/messages in Debug builds
#else
static constexpr bool kEnableValidation = false; // Disable in Release
#endif

// Convenience macro: check VkResult and abort early on error to simplify the sample
#define VK_CHECK(x)                                                                                    \
    do                                                                                                 \
    {                                                                                                  \
        VkResult err = (x);                                                                            \
        if (err != VK_SUCCESS)                                                                         \
        {                                                                                              \
            std::cerr << "Vulkan error " << err << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::abort();                                                                              \
        }                                                                                              \
    } while (0)

// Validation layer name used when kEnableValidation == true
static const char *kValidationLayer = "VK_LAYER_KHRONOS_validation";

// Function pointers for EXT_debug_utils. Loaded at runtime from the instance (they are extensions).
PFN_vkCreateDebugUtilsMessengerEXT pfnCreateDebugUtilsMessengerEXT = nullptr;
PFN_vkDestroyDebugUtilsMessengerEXT pfnDestroyDebugUtilsMessengerEXT = nullptr;

// Callback for validation and debug messages. Prints warnings/errors to stderr.
VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT /*types*/,
    const VkDebugUtilsMessengerCallbackDataEXT *callbackData,
    void * /*userData*/)
{
    // Filter to warning+error for signal-to-noise. Info/verbose are often too chatty for a sample.
    if (severity & (VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT))
    {
        std::cerr << "[Vulkan] " << callbackData->pMessage << std::endl;
    }
    return VK_FALSE; // Don't abort the Vulkan call, just report.
}

// Helper: check if a given validation layer is available on this system.
bool HasLayer(const char *name)
{
    uint32_t count = 0;
    VK_CHECK(vkEnumerateInstanceLayerProperties(&count, nullptr));
    std::vector<VkLayerProperties> props(count);
    VK_CHECK(vkEnumerateInstanceLayerProperties(&count, props.data()));
    for (auto &p : props)
        if (std::strcmp(p.layerName, name) == 0)
            return true;
    return false;
}

// Query GLFW for instance extensions it needs to create a window surface.
// Also tack on debug utils if validation is enabled and portability on macOS.
std::vector<const char *> GetInstanceExtensionsGLFW()
{
    uint32_t count = 0;
    const char **glfwExts = glfwGetRequiredInstanceExtensions(&count);
    std::vector<const char *> exts(glfwExts, glfwExts + count);
    if (kEnableValidation)
        exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#ifdef __APPLE__
    // MoltenVK requires enumeration portability on some setups.
    exts.push_back("VK_KHR_portability_enumeration");
#endif
    return exts;
}

// Load pointers to EXT_debug_utils functions from the instance.
void LoadDebugUtils(VkInstance instance)
{
    pfnCreateDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    pfnDestroyDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
}

// Create a debug messenger object if validation is enabled.
VkDebugUtilsMessengerEXT CreateDebugMessenger(VkInstance instance)
{
    if (!kEnableValidation || !pfnCreateDebugUtilsMessengerEXT)
        return VK_NULL_HANDLE;

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

// Queue family indices we care about: one for graphics, one for presenting.
// They may be the same family or different (e.g., on some systems).
struct QueueFamilyIndices
{
    std::optional<uint32_t> graphics;
    std::optional<uint32_t> present;
    bool complete() const { return graphics.has_value() && present.has_value(); }
};

// Find queue families on a physical device that support graphics and presentation to a given surface.
QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice pd, VkSurfaceKHR surface)
{
    QueueFamilyIndices indices;

    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, props.data());

    for (uint32_t i = 0; i < count; ++i)
    {
        if (props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            indices.graphics = i;

        VkBool32 presentSupport = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, surface, &presentSupport);
        if (presentSupport)
            indices.present = i;

        if (indices.complete())
            break;
    }
    return indices;
}

// Swapchain support details: capabilities + list of formats + present modes.
struct SwapchainSupport
{
    VkSurfaceCapabilitiesKHR caps{};            // min/max image count, current extent, transforms, etc.
    std::vector<VkSurfaceFormatKHR> formats;    // pixel formats + color spaces supported
    std::vector<VkPresentModeKHR> presentModes; // FIFO, MAILBOX, IMMEDIATE, etc.
};

// Query swapchain support for a device & surface.
SwapchainSupport QuerySwapchain(VkPhysicalDevice pd, VkSurfaceKHR surface)
{
    SwapchainSupport s{};

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pd, surface, &s.caps);

    uint32_t count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface, &count, nullptr);
    s.formats.resize(count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(pd, surface, &count, s.formats.data());

    count = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(pd, surface, &count, nullptr);
    s.presentModes.resize(count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(pd, surface, &count, s.presentModes.data());

    return s;
}

// Utility: choose a device memory type index that satisfies the requested properties.
uint32_t FindMemoryType(VkPhysicalDevice pd, uint32_t typeBits, VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(pd, &memProps);

    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
    {
        const bool typeAllowed = (typeBits & (1u << i)) != 0;
        const bool propsMatch = (memProps.memoryTypes[i].propertyFlags & props) == props;
        if (typeAllowed && propsMatch)
            return i;
    }
    throw std::runtime_error("No compatible memory type.");
}

// Minimal binary file reader (used to load precompiled SPIR-V shader modules).
std::vector<char> ReadFile(const std::string &path)
{
    FILE *f = fopen(path.c_str(), "rb");
    if (!f)
        throw std::runtime_error("Failed to open file: " + path);
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> data(size);
    fread(data.data(), 1, size, f);
    fclose(f);
    return data;
}

int main()
{
    // ---------------------------------------------------------------------------------
    // 0) Create a GLFW window (no OpenGL context; we'll use Vulkan)
    // ---------------------------------------------------------------------------------
    if (!glfwInit())
    {
        std::cerr << "GLFW init failed" << std::endl;
        return 1;
    }
    if (!glfwVulkanSupported())
    {
        std::cerr << "GLFW says Vulkan unsupported" << std::endl;
        return 1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // Don't create an OpenGL context
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);    // Allow resizing (we handle swapchain recreation)

    const int WIDTH = 800, HEIGHT = 600;
    GLFWwindow *window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Quad", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "Failed to create GLFW window" << std::endl;
        return 1;
    }

    // ---------------------------------------------------------------------------------
    // 1) Create a Vulkan instance (optionally with validation layers)
    // ---------------------------------------------------------------------------------
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan GLFW Quad";
    appInfo.apiVersion = VK_API_VERSION_1_2; // Request Vulkan 1.2 features/behavior for this sample

    std::vector<const char *> exts = GetInstanceExtensionsGLFW();
    std::vector<const char *> layers;
    if (kEnableValidation && HasLayer(kValidationLayer))
        layers.push_back(kValidationLayer);

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &appInfo;
    ici.enabledExtensionCount = (uint32_t)exts.size();
    ici.ppEnabledExtensionNames = exts.data();
    ici.enabledLayerCount = (uint32_t)layers.size();
    ici.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();
#ifdef __APPLE__
    // Required by VK_KHR_portability_enumeration when using MoltenVK on macOS.
    ici.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

    VkInstance instance{};
    VK_CHECK(vkCreateInstance(&ici, nullptr, &instance));

    // Load and create debug messenger (if enabled)
    LoadDebugUtils(instance);
    VkDebugUtilsMessengerEXT dbg = CreateDebugMessenger(instance);

    // ---------------------------------------------------------------------------------
    // 2) Create a window surface for presentation
    // ---------------------------------------------------------------------------------
    VkSurfaceKHR surface{};
    VK_CHECK(glfwCreateWindowSurface(instance, window, nullptr, &surface));

    // ---------------------------------------------------------------------------------
    // 3) Pick a physical device (GPU) + find queue families that support graphics/present
    // ---------------------------------------------------------------------------------
    uint32_t pdCount = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &pdCount, nullptr));
    if (pdCount == 0)
    {
        std::cerr << "No Vulkan devices" << std::endl;
        return 1;
    }

    std::vector<VkPhysicalDevice> devs(pdCount);
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &pdCount, devs.data()));

    VkPhysicalDevice phys = VK_NULL_HANDLE;
    QueueFamilyIndices qfi{};
    for (auto d : devs)
    {
        qfi = FindQueueFamilies(d, surface);
        if (qfi.complete())
        {
            phys = d; // Pick the first device that satisfies our needs (simple heuristic)
            break;
        }
    }
    if (phys == VK_NULL_HANDLE)
    {
        std::cerr << "No device with graphics+present" << std::endl;
        return 1;
    }

    // ---------------------------------------------------------------------------------
    // 4) Create a logical device + retrieve graphics & present queues
    // ---------------------------------------------------------------------------------
    float prio = 1.0f; // Highest priority for our single queue per family

    std::vector<VkDeviceQueueCreateInfo> qcis;
    std::vector<uint32_t> uniqueFamilies;
    uniqueFamilies.push_back(qfi.graphics.value());
    if (qfi.present.value() != qfi.graphics.value())
        uniqueFamilies.push_back(qfi.present.value());

    for (uint32_t fam : uniqueFamilies)
    {
        VkDeviceQueueCreateInfo qi{};
        qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi.queueFamilyIndex = fam;
        qi.queueCount = 1; // One queue per needed family is enough here
        qi.pQueuePriorities = &prio;
        qcis.push_back(qi);
    }

    // Device extensions we need. VK_KHR_swapchain is required for presenting to the window.
    const char *deviceExts[] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
#ifdef __APPLE__
        ,
        "VK_KHR_portability_subset" // Needed for portability surface/limits under MoltenVK
#endif
    };

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = (uint32_t)qcis.size();
    dci.pQueueCreateInfos = qcis.data();
    dci.enabledExtensionCount = sizeof(deviceExts) / sizeof(deviceExts[0]);
    dci.ppEnabledExtensionNames = deviceExts;

    VkDevice device{};
    VK_CHECK(vkCreateDevice(phys, &dci, nullptr, &device));

    // Grab queue handles from the created device
    VkQueue gfxQ{};
    VkQueue presentQ{};
    vkGetDeviceQueue(device, qfi.graphics.value(), 0, &gfxQ);
    vkGetDeviceQueue(device, qfi.present.value(), 0, &presentQ);

    // ---------------------------------------------------------------------------------
    // 5) Swapchain helper lambdas (format/present-mode selection)
    // ---------------------------------------------------------------------------------
    auto chooseSurfaceFormat = [&](const std::vector<VkSurfaceFormatKHR> &fmts)
    {
        // Prefer BGRA8 UNORM + sRGB nonlinear color space (common on desktop)
        for (auto f : fmts)
            if (f.format == VK_FORMAT_B8G8R8A8_UNORM && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                return f;
        return fmts[0]; // Fallback to the first available
    };

    auto choosePresentMode = [&](const std::vector<VkPresentModeKHR> &modes)
    {
        // Prefer MAILBOX (low latency / tear-free triple buffering) if available
        for (auto m : modes)
            if (m == VK_PRESENT_MODE_MAILBOX_KHR)
                return m;
        return VK_PRESENT_MODE_FIFO_KHR; // FIFO is guaranteed to be supported
    };

    // ---------------------------------------------------------------------------------
    // 6) Create a command pool for allocating primary command buffers
    // ---------------------------------------------------------------------------------
    VkCommandPoolCreateInfo poolCI{};
    poolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolCI.queueFamilyIndex = qfi.graphics.value();
    poolCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; // allow re-recording

    VkCommandPool cmdPool{};
    VK_CHECK(vkCreateCommandPool(device, &poolCI, nullptr, &cmdPool));

    // ---------------------------------------------------------------------------------
    // 7) Create pipeline resources that are independent of the swapchain
    //     - Vertex/index buffers (host visible for simplicity)
    //     - Shader modules
    //     - Vertex input state descriptions
    // ---------------------------------------------------------------------------------
    struct Vertex
    {
        float pos[2]; // clip-space XY
        float col[3]; // RGB color per-vertex
    };

    // A unit quad centered at origin, with per-vertex colors (will be interpolated)
    const Vertex verts[] = {
        {{-0.5f, -0.5f}, {1.f, 0.f, 0.f}},
        {{0.5f, -0.5f}, {0.f, 1.f, 0.f}},
        {{0.5f, 0.5f}, {0.f, 0.f, 1.f}},
        {{-0.5f, 0.5f}, {1.f, 1.f, 0.f}},
    };
    const uint16_t idx[] = {0, 1, 2, 2, 3, 0}; // two triangles

    // Helper to create a buffer + allocate/bind its memory
    auto createBuffer = [&](VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props, VkBuffer &buf, VkDeviceMemory &mem)
    {
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = size;
        bi.usage = usage;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(device, &bi, nullptr, &buf));

        VkMemoryRequirements mr{};
        vkGetBufferMemoryRequirements(device, buf, &mr);

        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = FindMemoryType(phys, mr.memoryTypeBits, props);
        VK_CHECK(vkAllocateMemory(device, &ai, nullptr, &mem));
        VK_CHECK(vkBindBufferMemory(device, buf, mem, 0));
    };

    // Vertex buffer (host visible to keep sample small; in real apps, stage to DEVICE_LOCAL)
    VkBuffer vbo{};
    VkDeviceMemory vmem{};
    createBuffer(sizeof(verts), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 vbo, vmem);

    // Index buffer (also host visible for simplicity)
    VkBuffer ibo{};
    VkDeviceMemory imem{};
    createBuffer(sizeof(idx), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 ibo, imem);

    // Upload CPU data into the mapped buffers
    void *p = nullptr;
    vkMapMemory(device, vmem, 0, sizeof(verts), 0, &p);
    std::memcpy(p, verts, sizeof(verts));
    vkUnmapMemory(device, vmem);

    vkMapMemory(device, imem, 0, sizeof(idx), 0, &p);
    std::memcpy(p, idx, sizeof(idx));
    vkUnmapMemory(device, imem);

    // Load SPIR-V shader modules (expects files to be present in working dir)
    auto vertCode = ReadFile("quad.vert.spv");
    auto fragCode = ReadFile("quad.frag.spv");

    auto makeShader = [&](const std::vector<char> &code)
    {
        VkShaderModuleCreateInfo sm{};
        sm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm.codeSize = code.size();
        sm.pCode = reinterpret_cast<const uint32_t *>(code.data());
        VkShaderModule m{};
        VK_CHECK(vkCreateShaderModule(device, &sm, nullptr, &m));
        return m;
    };
    VkShaderModule vs = makeShader(vertCode);
    VkShaderModule fs = makeShader(fragCode);

    // Describe vertex layout for the pipeline (binding + attributes)
    VkVertexInputBindingDescription bind{};
    bind.binding = 0;
    bind.stride = sizeof(Vertex);
    bind.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; // one per-vertex

    VkVertexInputAttributeDescription attrs[2]{};
    attrs[0].location = 0; // layout(location=0) in shader
    attrs[0].binding = 0;
    attrs[0].format = VK_FORMAT_R32G32_SFLOAT; // vec2 position
    attrs[0].offset = offsetof(Vertex, pos);

    attrs[1].location = 1; // layout(location=1) in shader
    attrs[1].binding = 0;
    attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT; // vec3 color
    attrs[1].offset = offsetof(Vertex, col);

    // ---------------------------------------------------------------------------------
    // 8) Swapchain + pipeline objects (recreated on resize)
    //     Everything below depends on the swapchain extent/format, so we store them
    //     and rebuild when the window is resized or swapchain becomes suboptimal.
    // ---------------------------------------------------------------------------------
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat swapFormat = VK_FORMAT_B8G8R8A8_UNORM; // default; actual chosen from surface
    VkExtent2D swapExtent{(uint32_t)WIDTH, (uint32_t)HEIGHT};

    std::vector<VkImage> swapImages;    // images owned by the swapchain
    std::vector<VkImageView> swapViews; // per-image view for rendering

    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    std::vector<VkFramebuffer> framebuffers; // one framebuffer per swapchain image
    std::vector<VkCommandBuffer> cmdBufs;    // pre-recorded draw commands per image

    // Destroy all objects tied to the current swapchain
    auto destroySwapchainObjects = [&]()
    {
        if (!device)
            return;

        for (auto fb : framebuffers)
            vkDestroyFramebuffer(device, fb, nullptr);
        framebuffers.clear();

        if (pipeline)
            vkDestroyPipeline(device, pipeline, nullptr), pipeline = VK_NULL_HANDLE;
        if (pipelineLayout)
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr), pipelineLayout = VK_NULL_HANDLE;
        if (renderPass)
            vkDestroyRenderPass(device, renderPass, nullptr), renderPass = VK_NULL_HANDLE;

        for (auto v : swapViews)
            vkDestroyImageView(device, v, nullptr);
        swapViews.clear();

        if (swapchain)
            vkDestroySwapchainKHR(device, swapchain, nullptr), swapchain = VK_NULL_HANDLE;

        if (!cmdBufs.empty())
        {
            vkFreeCommandBuffers(device, cmdPool, (uint32_t)cmdBufs.size(), cmdBufs.data());
            cmdBufs.clear();
        }
    };

    // Create swapchain and all dependent objects; also record command buffers.
    auto createSwapchainObjects = [&]()
    {
        // --- Query surface capabilities and choose settings ---
        auto supp = QuerySwapchain(phys, surface);
        VkSurfaceFormatKHR fmt = chooseSurfaceFormat(supp.formats);
        VkPresentModeKHR pmode = choosePresentMode(supp.presentModes);

        // Extent: if currentExtent is not UINT32_MAX, the surface size is fixed by the windowing system
        VkExtent2D extent{};
        if (supp.caps.currentExtent.width != UINT32_MAX)
            extent = supp.caps.currentExtent;
        else
        {
            int w, h;
            glfwGetFramebufferSize(window, &w, &h);
            extent = {(uint32_t)std::max(w, 1), (uint32_t)std::max(h, 1)};
        }

        // Choose number of images (one more than minimum for triple-like buffering if possible)
        uint32_t imageCount = supp.caps.minImageCount + 1;
        if (supp.caps.maxImageCount > 0 && imageCount > supp.caps.maxImageCount)
            imageCount = supp.caps.maxImageCount;

        // --- Create swapchain ---
        VkSwapchainCreateInfoKHR sci{};
        sci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        sci.surface = surface;
        sci.minImageCount = imageCount;
        sci.imageFormat = fmt.format;
        sci.imageColorSpace = fmt.colorSpace;
        sci.imageExtent = extent;
        sci.imageArrayLayers = 1;
        sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; // render directly into them

        uint32_t qIdx[2] = {qfi.graphics.value(), qfi.present.value()};
        if (qfi.graphics.value() != qfi.present.value())
        {
            // If graphics and present are separate families, use CONCURRENT sharing
            sci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            sci.queueFamilyIndexCount = 2;
            sci.pQueueFamilyIndices = qIdx;
        }
        else
        {
            // Otherwise EXCLUSIVE is more efficient
            sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        sci.preTransform = supp.caps.currentTransform; // No extra transform
        sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        sci.presentMode = pmode;
        sci.clipped = VK_TRUE; // OK to discard obscured pixels
        sci.oldSwapchain = VK_NULL_HANDLE;
        VK_CHECK(vkCreateSwapchainKHR(device, &sci, nullptr, &swapchain));

        swapFormat = fmt.format;
        swapExtent = extent;

        // --- Retrieve swapchain images and create image views ---
        uint32_t n = 0;
        vkGetSwapchainImagesKHR(device, swapchain, &n, nullptr);
        swapImages.resize(n);
        vkGetSwapchainImagesKHR(device, swapchain, &n, swapImages.data());

        swapViews.resize(n);
        for (uint32_t i = 0; i < n; ++i)
        {
            VkImageViewCreateInfo iv{};
            iv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            iv.image = swapImages[i];
            iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
            iv.format = swapFormat;
            iv.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VK_CHECK(vkCreateImageView(device, &iv, nullptr, &swapViews[i]));
        }

        // --- Render pass (single color attachment, no depth/stencil) ---
        VkAttachmentDescription color{};
        color.format = swapFormat;
        color.samples = VK_SAMPLE_COUNT_1_BIT;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;     // don't preserve previous
        color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; // ready for presentation

        VkAttachmentReference colorRef{};
        colorRef.attachment = 0;
        colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription sub{};
        sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.colorAttachmentCount = 1;
        sub.pColorAttachments = &colorRef;

        // External dependency: ensure color attachment writes happen at the right time
        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass = 0;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.srcAccessMask = 0;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rp.attachmentCount = 1;
        rp.pAttachments = &color;
        rp.subpassCount = 1;
        rp.pSubpasses = &sub;
        rp.dependencyCount = 1;
        rp.pDependencies = &dep;
        VK_CHECK(vkCreateRenderPass(device, &rp, nullptr, &renderPass));

        // --- Graphics pipeline (fixed state) ---
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vs;
        stages[0].pName = "main";

        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fs;
        stages[1].pName = "main";

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &bind;
        vi.vertexAttributeDescriptionCount = 2;
        vi.pVertexAttributeDescriptions = attrs;

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        // Static viewport/scissor matching the swapExtent
        VkViewport vp{};
        vp.x = 0;
        vp.y = 0;
        vp.width = (float)swapExtent.width;
        vp.height = (float)swapExtent.height;
        vp.minDepth = 0.f;
        vp.maxDepth = 1.f;

        VkRect2D sc{};
        sc.offset = {0, 0};
        sc.extent = swapExtent;

        VkPipelineViewportStateCreateInfo vpState{};
        vpState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vpState.viewportCount = 1;
        vpState.pViewports = &vp;
        vpState.scissorCount = 1;
        vpState.pScissors = &sc;

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE; // draw both faces (quad is CCW but keep simple)
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState cbAtt{};
        cbAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                               VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        cbAtt.blendEnable = VK_FALSE; // no blending needed

        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1;
        cb.pAttachments = &cbAtt;

        VkPipelineLayoutCreateInfo pl{};
        pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO; // No descriptors/push-constants
        VK_CHECK(vkCreatePipelineLayout(device, &pl, nullptr, &pipelineLayout));

        VkGraphicsPipelineCreateInfo gp{};
        gp.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gp.stageCount = 2;
        gp.pStages = stages;
        gp.pVertexInputState = &vi;
        gp.pInputAssemblyState = &ia;
        gp.pViewportState = &vpState;
        gp.pRasterizationState = &rs;
        gp.pMultisampleState = &ms;
        gp.pColorBlendState = &cb;
        gp.layout = pipelineLayout;
        gp.renderPass = renderPass;
        gp.subpass = 0;
        VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gp, nullptr, &pipeline));

        // --- Framebuffers (one per swapchain image view) ---
        framebuffers.resize(swapViews.size());
        for (size_t i = 0; i < swapViews.size(); ++i)
        {
            VkFramebufferCreateInfo f{};
            f.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            f.renderPass = renderPass;
            f.attachmentCount = 1;
            VkImageView att = swapViews[i];
            f.pAttachments = &att;
            f.width = swapExtent.width;
            f.height = swapExtent.height;
            f.layers = 1;
            VK_CHECK(vkCreateFramebuffer(device, &f, nullptr, &framebuffers[i]));
        }

        // --- Command buffers (record once per swapchain image) ---
        cmdBufs.resize(swapImages.size());

        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = cmdPool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = (uint32_t)cmdBufs.size();
        VK_CHECK(vkAllocateCommandBuffers(device, &ai, cmdBufs.data()));

        for (size_t i = 0; i < cmdBufs.size(); ++i)
        {
            VkCommandBufferBeginInfo begin{};
            begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            VK_CHECK(vkBeginCommandBuffer(cmdBufs[i], &begin));

            VkClearValue clear{}; // pleasant dark background
            clear.color = {{0.15f, 0.18f, 0.22f, 1.0f}};

            VkRenderPassBeginInfo rp{};
            rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            rp.renderPass = renderPass;
            rp.framebuffer = framebuffers[i];
            rp.renderArea = {{0, 0}, swapExtent};
            rp.clearValueCount = 1;
            rp.pClearValues = &clear;

            vkCmdBeginRenderPass(cmdBufs[i], &rp, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindPipeline(cmdBufs[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

            VkDeviceSize off = 0;
            vkCmdBindVertexBuffers(cmdBufs[i], 0, 1, &vbo, &off);
            vkCmdBindIndexBuffer(cmdBufs[i], ibo, 0, VK_INDEX_TYPE_UINT16);

            vkCmdDrawIndexed(cmdBufs[i], 6, 1, 0, 0, 0);
            vkCmdEndRenderPass(cmdBufs[i]);

            VK_CHECK(vkEndCommandBuffer(cmdBufs[i]));
        }
    };

    // Helper: Recreate swapchain when window is minimized/restored or resized.
    auto recreateSwapchain = [&]()
    {
        int w = 0, h = 0;
        // If minimized, width/height can be 0; wait until non-zero
        do
        {
            glfwGetFramebufferSize(window, &w, &h);
            glfwPollEvents();
        } while (w == 0 || h == 0);

        vkDeviceWaitIdle(device);  // ensure device not using old swapchain
        destroySwapchainObjects(); // free old resources
        createSwapchainObjects();  // build new ones for the new size
    };

    // Initial creation
    createSwapchainObjects();

    // ---------------------------------------------------------------------------------
    // 9) Synchronization objects used for frame pacing and acquire/present
    // ---------------------------------------------------------------------------------
    const int MAX_FRAMES = 2; // double-buffer CPU-GPU synchronization

    std::vector<VkSemaphore> imgAvail(MAX_FRAMES), renderDone(MAX_FRAMES);
    std::vector<VkFence> inFlight(MAX_FRAMES);

    VkSemaphoreCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (int i = 0; i < MAX_FRAMES; ++i)
    {
        VK_CHECK(vkCreateSemaphore(device, &sci, nullptr, &imgAvail[i]));
        VK_CHECK(vkCreateSemaphore(device, &sci, nullptr, &renderDone[i]));
        VK_CHECK(vkCreateFence(device, &fci, nullptr, &inFlight[i]));
    }

    // ---------------------------------------------------------------------------------
    // 10) Main render loop
    //     High-level per-frame flow:
    //       a) Wait for per-frame fence (CPU-GPU sync)
    //       b) Acquire next swapchain image → signal imgAvail[frame]
    //       c) Submit recorded command buffer → wait on imgAvail, signal renderDone
    //       d) Present → wait on renderDone
    // ---------------------------------------------------------------------------------
    uint32_t frame = 0; // index into per-frame sync objects

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // Helper: wait for a fence while keeping the window responsive
        auto waitFenceResponsive = [&](VkFence f)
        {
            for (;;)
            {
                VkResult r = vkWaitForFences(device, 1, &f, VK_TRUE, 0); // 0 = poll
                if (r == VK_SUCCESS)
                    return;
                if (r != VK_TIMEOUT)
                    VK_CHECK(r); // real error
                // keep UI alive while we wait
                glfwPollEvents();
            }
        };

        // Wait until the GPU has finished rendering the previous frame using this fence

        waitFenceResponsive(inFlight[frame]);
        VK_CHECK(vkResetFences(device, 1, &inFlight[frame]));

        // Acquire an image from the swapchain
        uint32_t imgIdx = 0;
        VkResult acq = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, imgAvail[frame], VK_NULL_HANDLE, &imgIdx);
        if (acq == VK_ERROR_OUT_OF_DATE_KHR)
        {
            // Swapchain no longer matches the window (resize/alt-tab). Recreate and retry.
            recreateSwapchain();
            continue;
        }
        else if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR)
        {
            std::cerr << "Acquire failed: " << acq << "\n";
            break;
        }

        // Submit draw commands; wait for the image-available semaphore before writing the image
        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &imgAvail[frame];
        si.pWaitDstStageMask = &waitStage; // at which pipeline stage to wait

        si.commandBufferCount = 1;
        VkCommandBuffer cb = cmdBufs[imgIdx];
        si.pCommandBuffers = &cb;

        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &renderDone[frame];

        VK_CHECK(vkQueueSubmit(gfxQ, 1, &si, inFlight[frame])); // fence will signal when GPU is done

        // Present the rendered image to the screen
        VkPresentInfoKHR pi{};
        pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &renderDone[frame];
        pi.swapchainCount = 1;
        pi.pSwapchains = &swapchain;
        pi.pImageIndices = &imgIdx;

        VkResult pres = vkQueuePresentKHR(presentQ, &pi);
        if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR)
        {
            // Window resized or surface changed; rebuild swapchain resources
            recreateSwapchain();
        }
        else if (pres != VK_SUCCESS)
        {
            std::cerr << "Present failed: " << pres << "\n";
            break;
        }

        // Advance to next frame (modulo number of in-flight frames)
        frame = (frame + 1) % MAX_FRAMES;
    }

    // Ensure GPU is idle before destruction to avoid tearing down in-use resources
    vkDeviceWaitIdle(device);

    // ---------------------------------------------------------------------------------
    // Cleanup (reverse order of creation where relevant)
    // ---------------------------------------------------------------------------------
    for (int i = 0; i < MAX_FRAMES; ++i)
    {
        vkDestroyFence(device, inFlight[i], nullptr);
        vkDestroySemaphore(device, renderDone[i], nullptr);
        vkDestroySemaphore(device, imgAvail[i], nullptr);
    }

    destroySwapchainObjects();

    // Buffers + memory
    vkDestroyBuffer(device, ibo, nullptr);
    vkFreeMemory(device, imem, nullptr);
    vkDestroyBuffer(device, vbo, nullptr);
    vkFreeMemory(device, vmem, nullptr);

    // Command pool & logical device
    vkDestroyCommandPool(device, cmdPool, nullptr);
    vkDestroyDevice(device, nullptr);

    // Instance-level objects
    vkDestroySurfaceKHR(instance, surface, nullptr);
    if (dbg && pfnDestroyDebugUtilsMessengerEXT)
        pfnDestroyDebugUtilsMessengerEXT(instance, dbg, nullptr);
    vkDestroyInstance(instance, nullptr);

    // GLFW window + termination
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}