# experiment.vulkan.hello\_world

Tiny, focused Vulkan experiments for learning-by-doing. Each folder is a 
self‑contained step with minimal code and lots of comments. This repo
currently uses **per‑folder Makefiles** (no top‑level build).

| Experiment       | Goal                                                                                                                                                           | Output                                                                                   |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **01.minimal**   | Stand up the absolute minimum — create **VkInstance**, pick **VkPhysicalDevice**, create **VkDevice**/**VkQueue**, and clean shutdown.                         | Console logs of adapters/queues; no window, no images.                                   |
| **02.offscreen** | Render without a window. Create an offscreen **VkImage** + memory/view/framebuffer, record a tiny pipeline + command buffer, submit, and write pixels to disk. | An image file written to the folder (SPIR-V is built from the GLSL shaders via `glslc`). |
| **03.onscreen**  | First triangle on screen. Create a surface & swapchain, render pass, graphics pipeline, per-frame synchronization, and present.                                | A window that shows a triangle; handles resize by recreating the swapchain.              |

## Prerequisites

* **Vulkan‑capable GPU** with up‑to‑date drivers
* **LunarG Vulkan SDK** installed (provides loader, validation layer, and **`glslc`** shader compiler)
* **C++20** compiler (Clang/GCC/MSVC)
* **Make** (per‑folder Makefiles)
* For on‑screen rendering, a windowing stack (e.g., GLFW) is linked by the Makefiles if present in the SDK/pkg‑config on your platform

## Build & run

Each experiment is built independently using its local **Makefile**.

```bash
# From the repo root, pick an experiment.
# eg.

cd 01.minimal
make           # builds the example
./executable      # name may vary per folder; see the Makefile output
```
