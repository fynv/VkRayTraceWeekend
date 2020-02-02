#include "volk.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <utility>
#include <unordered_map>
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>

#ifndef PI
#define PI 3.1415926f
#endif

struct BufferResource
{
	VkDeviceSize size;
	VkBuffer buf;
	VkDeviceMemory mem;
};

struct CommandBufferResource
{
	VkCommandBuffer buf;
};


static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData)
{
	printf("validation layer: %s\n", pCallbackData->pMessage);
	return VK_FALSE;
}

class Context
{
public:
	static Context& get_context()
	{
		static Context ctx;
		return ctx;
	}

	void buffer_create(BufferResource& buffer, VkDeviceSize size, bool ext_mem = false) const
	{
		buffer.size = size;
		if (size > 0)
		{
			if (ext_mem)
				_allocate_buffer_ex(buffer.buf, buffer.mem, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_EXT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			else
				_allocate_buffer(buffer.buf, buffer.mem, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_EXT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		}
	}

	void buffer_upload(BufferResource& buffer, const void* hdata) const
	{
		if (buffer.size == 0) return;
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		_allocate_buffer(stagingBuffer, stagingBufferMemory, buffer.size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		void* data;
		vkMapMemory(m_device, stagingBufferMemory, 0, buffer.size, 0, &data);
		memcpy(data, hdata, (size_t)buffer.size);
		vkUnmapMemory(m_device, stagingBufferMemory);

		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = m_commandPool_graphics;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(m_device, &allocInfo, &commandBuffer);

		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(commandBuffer, &beginInfo);
		VkBufferCopy copyRegion = {};
		copyRegion.srcOffset = 0;
		copyRegion.dstOffset = 0;
		copyRegion.size = buffer.size;
		vkCmdCopyBuffer(commandBuffer, stagingBuffer, buffer.buf, 1, &copyRegion);
		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(m_graphicsQueue);

		vkFreeCommandBuffers(m_device, m_commandPool_graphics, 1, &commandBuffer);
		_release_buffer(stagingBuffer, stagingBufferMemory);
	}

	void buffer_zero(BufferResource& buffer) const
	{
		if (buffer.size == 0) return;
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		_allocate_buffer(stagingBuffer, stagingBufferMemory, buffer.size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		void* data;
		vkMapMemory(m_device, stagingBufferMemory, 0, buffer.size, 0, &data);
		memset(data, 0, (size_t)buffer.size);
		vkUnmapMemory(m_device, stagingBufferMemory);

		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = m_commandPool_graphics;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(m_device, &allocInfo, &commandBuffer);

		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(commandBuffer, &beginInfo);
		VkBufferCopy copyRegion = {};
		copyRegion.srcOffset = 0;
		copyRegion.dstOffset = 0;
		copyRegion.size = buffer.size;
		vkCmdCopyBuffer(commandBuffer, stagingBuffer, buffer.buf, 1, &copyRegion);
		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(m_graphicsQueue);

		vkFreeCommandBuffers(m_device, m_commandPool_graphics, 1, &commandBuffer);
		_release_buffer(stagingBuffer, stagingBufferMemory);;

	}

	void buffer_download(const BufferResource& buffer, void* hdata, VkDeviceSize begin = 0, VkDeviceSize end = (VkDeviceSize)(-1))
	{
		if (end > buffer.size) end = buffer.size;
		if (end <= begin) return;

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		_allocate_buffer(stagingBuffer, stagingBufferMemory, end - begin, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = m_commandPool_graphics;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(m_device, &allocInfo, &commandBuffer);

		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(commandBuffer, &beginInfo);
		VkBufferCopy copyRegion = {};
		copyRegion.srcOffset = begin;
		copyRegion.dstOffset = 0;
		copyRegion.size = end - begin;
		vkCmdCopyBuffer(commandBuffer, buffer.buf, stagingBuffer, 1, &copyRegion);
		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(m_graphicsQueue);

		void* data;
		vkMapMemory(m_device, stagingBufferMemory, 0, end - begin, 0, &data);
		memcpy(hdata, data, (size_t)(end - begin));
		vkUnmapMemory(m_device, stagingBufferMemory);

		vkFreeCommandBuffers(m_device, m_commandPool_graphics, 1, &commandBuffer);
		_release_buffer(stagingBuffer, stagingBufferMemory);

	}

	uint64_t buffer_get_device_address(const BufferResource& buffer) const
	{
		if (buffer.size == 0) return 0;
		VkBufferDeviceAddressInfoEXT bufAdrInfo = {};
		bufAdrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_EXT;
		bufAdrInfo.buffer = buffer.buf;
		return vkGetBufferDeviceAddressEXT(m_device, &bufAdrInfo);
	}

	void buffer_release(BufferResource& buffer) const
	{
		if (buffer.size > 0)
			_release_buffer(buffer.buf, buffer.mem);
	}

	void command_buffer_create(CommandBufferResource& cmdBuf, bool one_time_submit = false) const
	{
		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = m_commandPool_graphics;
		allocInfo.commandBufferCount = 1;

		vkAllocateCommandBuffers(m_device, &allocInfo, &cmdBuf.buf);

		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		if (one_time_submit)
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	
		vkBeginCommandBuffer(cmdBuf.buf, &beginInfo);
	}

	void command_buffer_release(CommandBufferResource& cmdBuf) const
	{
		vkFreeCommandBuffers(m_device, m_commandPool_graphics, 1, &cmdBuf.buf);
	}

	void queue_wait() const
	{
		vkQueueWaitIdle(m_graphicsQueue);
	}

	void queue_submit(CommandBufferResource& cmdBuf)
	{
		vkEndCommandBuffer(cmdBuf.buf);
		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &cmdBuf.buf;
		vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, 0);
	}

	VkDevice& device() { return m_device; }
	VkQueue& queue() { return m_graphicsQueue;  }
	VkPhysicalDeviceRayTracingPropertiesNV& raytracing_properties() { return m_raytracingProperties; }

	void _allocate_buffer(VkBuffer& buf, VkDeviceMemory& mem, size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags flags) const
	{
		if (size == 0) return;

		VkBufferCreateInfo bufferCreateInfo = {};
		bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferCreateInfo.size = size;
		bufferCreateInfo.usage = usage;
		bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		vkCreateBuffer(m_device, &bufferCreateInfo, nullptr, &buf);

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(m_device, buf, &memRequirements);

		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProperties);

		uint32_t memoryTypeIndex = VK_MAX_MEMORY_TYPES;
		for (uint32_t k = 0; k < memProperties.memoryTypeCount; k++)
		{
			if ((memRequirements.memoryTypeBits & (1 << k)) == 0) continue;
			if ((flags & memProperties.memoryTypes[k].propertyFlags) == flags)
			{
				memoryTypeIndex = k;
				break;
			}
		}

		VkMemoryAllocateInfo memoryAllocateInfo = {};
		memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memoryAllocateInfo.allocationSize = memRequirements.size;
		memoryAllocateInfo.memoryTypeIndex = memoryTypeIndex;

		vkAllocateMemory(m_device, &memoryAllocateInfo, nullptr, &mem);
		vkBindBufferMemory(m_device, buf, mem, 0);
	}

	void _allocate_buffer_ex(VkBuffer& buf, VkDeviceMemory& mem, size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags flags) const
	{
		if (size == 0) return;

		VkBufferCreateInfo bufferCreateInfo = {};
		bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferCreateInfo.size = size;
		bufferCreateInfo.usage = usage;
		bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		vkCreateBuffer(m_device, &bufferCreateInfo, nullptr, &buf);

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(m_device, buf, &memRequirements);

		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProperties);

		uint32_t memoryTypeIndex = VK_MAX_MEMORY_TYPES;
		for (uint32_t k = 0; k < memProperties.memoryTypeCount; k++)
		{
			if ((memRequirements.memoryTypeBits & (1 << k)) == 0) continue;
			if ((flags & memProperties.memoryTypes[k].propertyFlags) == flags)
			{
				memoryTypeIndex = k;
				break;
			}
		}

		VkExportMemoryAllocateInfoKHR vulkanExportMemoryAllocateInfoKHR = {};
		vulkanExportMemoryAllocateInfoKHR.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
#ifdef _WIN64
		vulkanExportMemoryAllocateInfoKHR.pNext = NULL;
		vulkanExportMemoryAllocateInfoKHR.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
#else
		vulkanExportMemoryAllocateInfoKHR.pNext = NULL;
		vulkanExportMemoryAllocateInfoKHR.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
		VkMemoryAllocateInfo memoryAllocateInfo = {};
		memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memoryAllocateInfo.pNext = &vulkanExportMemoryAllocateInfoKHR;
		memoryAllocateInfo.allocationSize = memRequirements.size;
		memoryAllocateInfo.memoryTypeIndex = memoryTypeIndex;

		vkAllocateMemory(m_device, &memoryAllocateInfo, nullptr, &mem);
		vkBindBufferMemory(m_device, buf, mem, 0);
	}

	void _release_buffer(VkBuffer& buf, VkDeviceMemory& mem) const
	{
		vkDestroyBuffer(m_device, buf, nullptr);
		vkFreeMemory(m_device, mem, nullptr);
	}

private:
	VkDebugUtilsMessengerEXT m_debugMessenger;
	VkInstance m_instance;
	VkPhysicalDevice m_physicalDevice;
	VkPhysicalDeviceBufferDeviceAddressFeaturesEXT m_bufferDeviceAddressFeatures;
	VkPhysicalDeviceFeatures2 m_features2;
	VkPhysicalDeviceRayTracingPropertiesNV m_raytracingProperties;
	uint32_t m_graphicsQueueFamily;
	float m_queuePriority;
	VkDevice m_device;
	VkQueue m_graphicsQueue;
	VkCommandPool m_commandPool_graphics;

	bool _init_vulkan()
	{
		if (volkInitialize() != VK_SUCCESS) return false;

		{
			VkApplicationInfo appInfo = {};
			appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
			appInfo.pApplicationName = "ThrustVk";
			appInfo.applicationVersion = VK_MAKE_VERSION(1, 1, 0);
			appInfo.pEngineName = "No Engine";
			appInfo.engineVersion = VK_MAKE_VERSION(1, 1, 0);
			appInfo.apiVersion = VK_API_VERSION_1_1;

			const char* name_extensions[] = { 
				VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
				VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
				VK_EXT_DEBUG_UTILS_EXTENSION_NAME
			};

			char str_validationLayers[] = "VK_LAYER_KHRONOS_validation";
			const char* validationLayers[] = { str_validationLayers };

			VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
			debugCreateInfo = {};
			debugCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
			debugCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
			debugCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
			debugCreateInfo.pfnUserCallback = debugCallback;

			VkInstanceCreateInfo createInfo = {};
			createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
			createInfo.pApplicationInfo = &appInfo;
			createInfo.enabledExtensionCount = 3;
			createInfo.ppEnabledExtensionNames = name_extensions;

#ifdef _DEBUG
			createInfo.enabledLayerCount = 1;
			createInfo.ppEnabledLayerNames = validationLayers;
			createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
#endif

			if (vkCreateInstance(&createInfo, nullptr, &m_instance) != VK_SUCCESS) return false;

#ifdef _DEBUG
			PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT");
			vkCreateDebugUtilsMessengerEXT(m_instance, &debugCreateInfo, nullptr, &m_debugMessenger);
#endif
		}
		volkLoadInstance(m_instance);

		m_physicalDevice = VK_NULL_HANDLE;
		{
			// select physical device
			uint32_t deviceCount = 0;
			vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr);
			std::vector<VkPhysicalDevice> ph_devices(deviceCount);
			vkEnumeratePhysicalDevices(m_instance, &deviceCount, ph_devices.data());
			m_physicalDevice = ph_devices[0];
		}

		m_bufferDeviceAddressFeatures = {};
		{
			m_bufferDeviceAddressFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_ADDRESS_FEATURES_EXT;
			m_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
			m_features2.pNext = &m_bufferDeviceAddressFeatures;
			m_features2.features = {};
			vkGetPhysicalDeviceFeatures2(m_physicalDevice, &m_features2);
		}


		m_raytracingProperties = {};
		{
			m_raytracingProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PROPERTIES_NV;
			VkPhysicalDeviceProperties2 props;
			props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
			props.pNext = &m_raytracingProperties;
			props.properties = {};
			vkGetPhysicalDeviceProperties2(m_physicalDevice, &props);
		}

		m_graphicsQueueFamily = (uint32_t)(-1);
		{
			uint32_t queueFamilyCount = 0;
			vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueFamilyCount, nullptr);
			std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
			vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueFamilyCount, queueFamilies.data());

			for (uint32_t i = 0; i < queueFamilyCount; i++)
				if (m_graphicsQueueFamily == (uint32_t)(-1) && queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) m_graphicsQueueFamily = i;
		}

		// logical device/queue
		m_queuePriority = 1.0f;

		{
			VkDeviceQueueCreateInfo queueCreateInfo = {};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = m_graphicsQueueFamily;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.pQueuePriorities = &m_queuePriority;

			const char* name_extensions[] = {				
				VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
				VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
			#ifdef _WIN64
				VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
			#else
				VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
			#endif
				VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
				VK_NV_RAY_TRACING_EXTENSION_NAME,		
			};

			VkDeviceCreateInfo createInfo = {};
			createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
			createInfo.pQueueCreateInfos = &queueCreateInfo;
			createInfo.queueCreateInfoCount = 1;
			createInfo.enabledExtensionCount = 5;
			createInfo.ppEnabledExtensionNames = name_extensions;
			createInfo.pNext = &m_features2;

			if (vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_device) != VK_SUCCESS) return false;
		}

		vkGetDeviceQueue(m_device, m_graphicsQueueFamily, 0, &m_graphicsQueue);

		{
			VkCommandPoolCreateInfo poolInfo = {};
			poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
			poolInfo.queueFamilyIndex = m_graphicsQueueFamily;
			vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_commandPool_graphics);
		}

		return true;

	}

	Context()
	{
		if (!_init_vulkan()) exit(0);
	}

	~Context()
	{
		vkDestroyCommandPool(m_device, m_commandPool_graphics, nullptr);
		vkDestroyDevice(m_device, nullptr);
		vkDestroyInstance(m_instance, nullptr);
	}
};

