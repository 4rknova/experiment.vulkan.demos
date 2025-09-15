// main.cpp
// A minimal Vulkan "Hello, Vulkan!" program that:
//  1) Creates a Vulkan instance (with validation + debug utils if enabled)
//  2) Picks a physical GPU and finds a graphics queue family
//  3) Creates a logical device and obtains a graphics queue
//  4) Allocates a command buffer, records a no-op, submits it, and waits
//  5) Prints the selected GPU name and cleans up all resources

#include <vulkan/vulkan.h>

#include <iostream>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <optional>
#include <cstdlib>

// Enable validation layers in debug builds only. In release builds, the
// validation layer and debug utils extension are disabled for performance.
#ifndef NDEBUG
static constexpr bool kEnableValidation = true;
#else
static constexpr bool kEnableValidation = false;
#endif

// Simple helper macro to check VkResult return codes. If a Vulkan call fails,
// it logs the error code and source location, then aborts the program.
// NOTE: For production code, consider mapping VkResult to readable strings.
#define VK_CHECK(x) do { VkResult err = (x); if (err != VK_SUCCESS) { \
    std::cerr << "Vulkan error " << err << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::abort(); }} while (0)

// The standard Khronos validation layer name.
static const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

// Function pointers for the VK_EXT_debug_utils extension entry points.
// These are instance-level extension functions and must be fetched with
// vkGetInstanceProcAddr after creating the instance.
PFN_vkCreateDebugUtilsMessengerEXT  pfnCreateDebugUtilsMessengerEXT  = nullptr;
PFN_vkDestroyDebugUtilsMessengerEXT pfnDestroyDebugUtilsMessengerEXT = nullptr;

// Debug messenger callback used by VK_EXT_debug_utils. Vulkan calls this
// function to report validation and performance messages.
// We print warnings and errors to stderr. Returning VK_FALSE indicates that
// validation should not abort the call that triggered the message.
VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT /*types*/,
    const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
    void* /*userData*/) {
    if (severity & (VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)) {
        std::cerr << "[Vulkan] " << callbackData->pMessage << std::endl;
    }
    return VK_FALSE;
}

// Returns true if the requested layer is available on this system.
// This is useful to avoid enabling validation on systems that don't have
// the Khronos validation layer installed.
bool HasLayer(const char* name) {
    uint32_t count = 0;
    VK_CHECK(vkEnumerateInstanceLayerProperties(&count, nullptr));
    std::vector<VkLayerProperties> props(count);
    VK_CHECK(vkEnumerateInstanceLayerProperties(&count, props.data()));
    for (auto& p : props) {
        if (std::strcmp(p.layerName, name) == 0) return true;
    }
    return false;
}

// Collect instance extensions required by this sample at runtime.
// - VK_EXT_debug_utils: needed to create the debug messenger when validation
//   is enabled (provides structured messages and labels).
// - On Apple (MoltenVK), VK_KHR_portability_enumeration is required to
//   enumerate "portable" physical devices.
std::vector<const char*> GetRequiredInstanceExtensions() {
    std::vector<const char*> exts;
    if (kEnableValidation) {
        exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
#ifdef __APPLE__
    // Required by MoltenVK to expose portability enumeration behavior.
    exts.push_back("VK_KHR_portability_enumeration");
#endif
    return exts;
}

// Loads the debug utils extension entry points from the created instance.
// Must be called after vkCreateInstance succeeds.
void LoadDebugUtils(VkInstance instance) {
    pfnCreateDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    pfnDestroyDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
}

// Creates a debug messenger if validation is enabled and the function pointer
// is available. The messenger will forward warnings and errors to DebugCallback.
VkDebugUtilsMessengerEXT CreateDebugMessenger(VkInstance instance) {
    if (!kEnableValidation || !pfnCreateDebugUtilsMessengerEXT) return VK_NULL_HANDLE;

    VkDebugUtilsMessengerCreateInfoEXT ci{};
    ci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    ci.pNext = nullptr;
    ci.flags = 0;
    // Report warning and error severities; you can also include INFO/VERBOSE
    // during development by OR-ing additional bits.
    ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    // Types of messages we care about.
    ci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    ci.pfnUserCallback = DebugCallback;
    ci.pUserData = nullptr;

    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    VK_CHECK(pfnCreateDebugUtilsMessengerEXT(instance, &ci, nullptr, &messenger));
    return messenger;
}

// Small helper to track indices of queue families we need.
// In this sample we only care about a graphics-capable queue.
struct QueueFamilyIndices {
    std::optional<uint32_t> graphics;
    bool complete() const { return graphics.has_value(); }
};

// Scans the physical device's queue families and picks one that supports
// graphics operations. Returns indices with 'graphics' set if found.
QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice pd) {
    QueueFamilyIndices indices;

    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &count, props.data());

    for (uint32_t i = 0; i < count; ++i) {
        if (props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphics = i;
            break;
        }
    }
    return indices;
}

// Chooses a suitable physical device (GPU).
// Preference order:
//   1) Discrete GPU with a graphics queue
//   2) Any GPU with a graphics queue
// Throws if no suitable device is found.
VkPhysicalDevice PickPhysicalDevice(VkInstance instance) {
    uint32_t count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, nullptr));
    if (count == 0) throw std::runtime_error("No Vulkan-capable GPUs found.");

    std::vector<VkPhysicalDevice> devices(count);
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, devices.data()));

    VkPhysicalDevice chosen = VK_NULL_HANDLE;

    // Try to find a discrete GPU first.
    for (auto d : devices) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(d, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
            FindQueueFamilies(d).complete()) {
            chosen = d;
            break;
        }
    }

    // Fallback: any GPU with graphics support.
    if (chosen == VK_NULL_HANDLE) {
        for (auto d : devices) {
            if (FindQueueFamilies(d).complete()) {
                chosen = d;
                break;
            }
        }
    }

    if (chosen == VK_NULL_HANDLE) {
        throw std::runtime_error("No suitable GPU with graphics queue.");
    }
    return chosen;
}

int main() {
    try {
        // =========================
        // 1) Create a Vulkan instance
        // =========================
        // The VkApplicationInfo provides optional app/engine identifiers and
        // the Vulkan API version you intend to use. Drivers may use this for
        // optimizations or compatibility behavior.
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Vulkan Hello World";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_2; // Request Vulkan 1.2 features

        // Enable validation layer if available and requested (debug builds).
        std::vector<const char*> layers;
        if (kEnableValidation && HasLayer(kValidationLayer)) {
            layers.push_back(kValidationLayer);
        }

        // Gather instance extensions (debug utils and portability on Apple).
        auto exts = GetRequiredInstanceExtensions();

        // Instance creation info
        VkInstanceCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pNext = nullptr;
        ici.flags = 0;
        ici.pApplicationInfo = &appInfo;
        ici.enabledLayerCount = static_cast<uint32_t>(layers.size());
        ici.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();
        ici.enabledExtensionCount = static_cast<uint32_t>(exts.size());
        ici.ppEnabledExtensionNames = exts.empty() ? nullptr : exts.data();
#ifdef __APPLE__
        // Required for VK_KHR_portability_enumeration to affect enumeration.
        ici.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

        VkInstance instance = VK_NULL_HANDLE;
        VK_CHECK(vkCreateInstance(&ici, nullptr, &instance));

        // Load extension function pointers now that we have an instance.
        LoadDebugUtils(instance);

        // Create a debug messenger for validation messages (if enabled).
        VkDebugUtilsMessengerEXT debugMessenger = CreateDebugMessenger(instance);

        // ===================================
        // 2) Select a physical device (GPU)
        // ===================================
        VkPhysicalDevice physical = PickPhysicalDevice(instance);

        // Query some properties for display/logging.
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physical, &props);

        // Find a queue family index that supports graphics.
        QueueFamilyIndices indices = FindQueueFamilies(physical);
        uint32_t graphicsFamily = indices.graphics.value();

        // ======================================
        // 3) Create a logical device + get queue
        // ======================================
        // Specify one queue from the graphics-capable family with highest priority.
        float priority = 1.0f;
        VkDeviceQueueCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = graphicsFamily;
        qci.queueCount = 1;
        qci.pQueuePriorities = &priority;

        // Device extensions: on Apple/MoltenVK, portability_subset is required.
        std::vector<const char*> deviceExts;
#ifdef __APPLE__
        deviceExts.push_back("VK_KHR_portability_subset");
#endif

        // We are not enabling any device features in this minimal sample.
        VkDeviceCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = static_cast<uint32_t>(deviceExts.size());
        dci.ppEnabledExtensionNames = deviceExts.empty() ? nullptr : deviceExts.data();
        dci.pEnabledFeatures = nullptr; // or point to VkPhysicalDeviceFeatures

        VkDevice device = VK_NULL_HANDLE;
        VK_CHECK(vkCreateDevice(physical, &dci, nullptr, &device));

        // Retrieve the VkQueue handle from the logical device.
        VkQueue graphicsQueue = VK_NULL_HANDLE;
        vkGetDeviceQueue(device, graphicsFamily, 0, &graphicsQueue);

        // =========================================
        // 4) Create a command pool and command buffer
        // =========================================
        // Command pools allocate and recycle command buffers for a given queue family.
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        // RESET flag allows individual command buffers to be reset.
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = graphicsFamily;

        VkCommandPool cmdPool = VK_NULL_HANDLE;
        VK_CHECK(vkCreateCommandPool(device, &poolInfo, nullptr, &cmdPool));

        // Allocate one primary command buffer from the pool.
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = cmdPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, &cmd));

        // Begin recording a minimal command buffer (no actual commands).
        // Even an empty command buffer must be begun/ended to be valid for submission.
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));
        VK_CHECK(vkEndCommandBuffer(cmd));

        // ==============================
        // 5) Submit the command + wait
        // ==============================
        // Create a fence to know when the GPU has finished executing our submission.
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

        VkFence fence = VK_NULL_HANDLE;
        VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &fence));

        // Package our command buffer in a VkSubmitInfo for queue submission.
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;

        // Submit to the graphics queue and signal the fence when done.
        VK_CHECK(vkQueueSubmit(graphicsQueue, 1, &submit, fence));

        // Wait up to 2 seconds for GPU to finish. (Nanoseconds)
        VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_C(2'000'000'000)));

        std::cout << "Hello, Vulkan! Running on: " << props.deviceName << std::endl;

        // =========
        // Cleanup
        // =========
        // Always destroy in reverse order of creation to respect dependencies.
        vkDestroyFence(device, fence, nullptr);
        vkFreeCommandBuffers(device, cmdPool, 1, &cmd);
        vkDestroyCommandPool(device, cmdPool, nullptr);
        vkDestroyDevice(device, nullptr);

        // Destroy the debug messenger if it was created.
        if (debugMessenger && pfnDestroyDebugUtilsMessengerEXT) {
            pfnDestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        // Finally, destroy the Vulkan instance.
        vkDestroyInstance(instance, nullptr);
        return 0;
    } catch (const std::exception& e) {
        // If anything throws (e.g., no GPU, failed creation), log and return failure.
        std::cerr << "Fatal: " << e.what() << std::endl;
        return 1;
    }
}