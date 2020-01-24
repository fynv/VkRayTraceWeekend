#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include "context.inl"
#include "PathTracer.h"

struct AccelerationResource
{
	VkBuffer scratchBuffer = VK_NULL_HANDLE;
	VkDeviceMemory scratchMem = VK_NULL_HANDLE;
	VkBuffer resultBuffer = VK_NULL_HANDLE;
	VkDeviceMemory resultMem = VK_NULL_HANDLE;
	VkBuffer instancesBuffer = VK_NULL_HANDLE;
	VkDeviceMemory instancesMem = VK_NULL_HANDLE;
	VkAccelerationStructureNV structure = VK_NULL_HANDLE;
};

void as_release(AccelerationResource* as)
{
	Context& ctx = Context::get_context();
	ctx._release_buffer(as->scratchBuffer, as->scratchMem);
	ctx._release_buffer(as->resultBuffer, as->resultMem);
	ctx._release_buffer(as->instancesBuffer, as->instancesMem);
	vkDestroyAccelerationStructureNV(ctx.device(), as->structure, nullptr);
}


Geometry::Geometry(const glm::mat4x4& model, glm::vec3 color)
{
	m_color = color;
	m_model = model;
	m_norm_mat = glm::transpose(glm::inverse(model));

	m_blas = new AccelerationResource;
}

Geometry::~Geometry()
{
	delete m_blas;
}

void TriangleMesh::_blas_create()
{
	Context& ctx = Context::get_context();

	VkGeometryNV geometry = {};
	geometry.sType = VK_STRUCTURE_TYPE_GEOMETRY_NV;
	geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_NV;
	geometry.geometry.triangles = {};
	geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_GEOMETRY_TRIANGLES_NV;
	geometry.geometry.triangles.vertexData = m_vertexBuffer->buf;
	geometry.geometry.triangles.vertexOffset = 0;
	geometry.geometry.triangles.vertexCount = (unsigned)(m_vertexBuffer->size / sizeof(Vertex));
	geometry.geometry.triangles.vertexStride = sizeof(Vertex);
	geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
	geometry.geometry.triangles.indexData = m_indexBuffer->buf;
	geometry.geometry.triangles.indexOffset = 0;
	geometry.geometry.triangles.indexCount = (unsigned)(m_indexBuffer->size / sizeof(unsigned));
	geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
	geometry.geometry.triangles.transformData = VK_NULL_HANDLE;
	geometry.geometry.triangles.transformOffset = 0;
	geometry.geometry.aabbs = { VK_STRUCTURE_TYPE_GEOMETRY_AABB_NV };
	geometry.flags = VK_GEOMETRY_OPAQUE_BIT_NV;

	VkAccelerationStructureInfoNV accelerationStructureInfo = {};
	accelerationStructureInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV;
	accelerationStructureInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV;
	accelerationStructureInfo.geometryCount = 1;
	accelerationStructureInfo.pGeometries = &geometry;

	VkAccelerationStructureCreateInfoNV accelerationStructureCreateInfo = {};
	accelerationStructureCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV;
	accelerationStructureCreateInfo.info = accelerationStructureInfo;

	vkCreateAccelerationStructureNV(ctx.device(), &accelerationStructureCreateInfo, nullptr, &m_blas->structure);

	VkDeviceSize scratchSizeInBytes = 0;
	VkDeviceSize resultSizeInBytes = 0;

	{
		VkAccelerationStructureMemoryRequirementsInfoNV memoryRequirementsInfo = {};
		memoryRequirementsInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV;
		memoryRequirementsInfo.accelerationStructure = m_blas->structure;

		VkMemoryRequirements2 memoryRequirements;
		memoryRequirementsInfo.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_NV;
		vkGetAccelerationStructureMemoryRequirementsNV(ctx.device(), &memoryRequirementsInfo, &memoryRequirements);
		resultSizeInBytes = memoryRequirements.memoryRequirements.size;
		memoryRequirementsInfo.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_NV;
		vkGetAccelerationStructureMemoryRequirementsNV(ctx.device(), &memoryRequirementsInfo, &memoryRequirements);
		scratchSizeInBytes = memoryRequirements.memoryRequirements.size;
		memoryRequirementsInfo.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_UPDATE_SCRATCH_NV;
		vkGetAccelerationStructureMemoryRequirementsNV(ctx.device(), &memoryRequirementsInfo, &memoryRequirements);
		if (memoryRequirements.memoryRequirements.size > scratchSizeInBytes) scratchSizeInBytes = memoryRequirements.memoryRequirements.size;
	}

	ctx._allocate_buffer(m_blas->scratchBuffer, m_blas->scratchMem, scratchSizeInBytes, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	ctx._allocate_buffer(m_blas->resultBuffer, m_blas->resultMem, resultSizeInBytes, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);


	CommandBufferResource cmdBuf;
	ctx.command_buffer_create(cmdBuf, true);

	{
		VkBindAccelerationStructureMemoryInfoNV bindInfo = {};
		bindInfo.sType = VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV;
		bindInfo.accelerationStructure = m_blas->structure;
		bindInfo.memory = m_blas->resultMem;

		vkBindAccelerationStructureMemoryNV(ctx.device(), 1, &bindInfo);

		vkCmdBuildAccelerationStructureNV(cmdBuf.buf, &accelerationStructureInfo, VK_NULL_HANDLE, 0, VK_FALSE,
			m_blas->structure, VK_NULL_HANDLE, m_blas->scratchBuffer, 0);
	}

	ctx.queue_submit(cmdBuf);
	ctx.queue_wait();
	ctx.command_buffer_release(cmdBuf);

}

TriangleMesh::TriangleMesh(const glm::mat4x4& model, const std::vector<Vertex>& vertices, const std::vector<unsigned>& indices, glm::vec3 color) : Geometry(model, color)
{
	m_vertexCount = (unsigned)vertices.size();
	m_indexCount = (unsigned)indices.size();

	m_vertexBuffer = new BufferResource;
	m_indexBuffer = new BufferResource;

	Context& ctx = Context::get_context();
	ctx.buffer_create(*m_vertexBuffer, sizeof(Vertex)* m_vertexCount);
	ctx.buffer_upload(*m_vertexBuffer, vertices.data());
	ctx.buffer_create(*m_indexBuffer, sizeof(unsigned)*m_indexCount);
	ctx.buffer_upload(*m_indexBuffer, indices.data());

	_blas_create();

}

TriangleMesh::~TriangleMesh()
{
	Context& ctx = Context::get_context();
	as_release(m_blas);
	ctx.buffer_release(*m_vertexBuffer);
	ctx.buffer_release(*m_indexBuffer);

	delete m_indexBuffer;
	delete m_vertexBuffer;
}

void UnitSphere::_blas_create()
{
	Context& ctx = Context::get_context();

	VkGeometryNV geometry = {};
	geometry.sType = VK_STRUCTURE_TYPE_GEOMETRY_NV;
	geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_NV;
	geometry.geometry.triangles = {};
	geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_GEOMETRY_TRIANGLES_NV;
	geometry.geometry.aabbs = {};
	geometry.geometry.aabbs.sType = VK_STRUCTURE_TYPE_GEOMETRY_AABB_NV;
	geometry.geometry.aabbs.aabbData = m_aabb_buf->buf;
	geometry.geometry.aabbs.offset = 0;
	geometry.geometry.aabbs.numAABBs = 1;
	geometry.geometry.aabbs.stride = 0;
	geometry.flags = VK_GEOMETRY_OPAQUE_BIT_NV;

	VkAccelerationStructureInfoNV accelerationStructureInfo = {};
	accelerationStructureInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV;
	accelerationStructureInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV;
	accelerationStructureInfo.geometryCount = 1;
	accelerationStructureInfo.pGeometries = &geometry;

	VkAccelerationStructureCreateInfoNV accelerationStructureCreateInfo = {};
	accelerationStructureCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV;
	accelerationStructureCreateInfo.info = accelerationStructureInfo;

	vkCreateAccelerationStructureNV(ctx.device(), &accelerationStructureCreateInfo, nullptr, &m_blas->structure);

	VkDeviceSize scratchSizeInBytes = 0;
	VkDeviceSize resultSizeInBytes = 0;

	{
		VkAccelerationStructureMemoryRequirementsInfoNV memoryRequirementsInfo = {};
		memoryRequirementsInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV;
		memoryRequirementsInfo.accelerationStructure = m_blas->structure;

		VkMemoryRequirements2 memoryRequirements;
		memoryRequirementsInfo.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_NV;
		vkGetAccelerationStructureMemoryRequirementsNV(ctx.device(), &memoryRequirementsInfo, &memoryRequirements);
		resultSizeInBytes = memoryRequirements.memoryRequirements.size;
		memoryRequirementsInfo.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_NV;
		vkGetAccelerationStructureMemoryRequirementsNV(ctx.device(), &memoryRequirementsInfo, &memoryRequirements);
		scratchSizeInBytes = memoryRequirements.memoryRequirements.size;
		memoryRequirementsInfo.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_UPDATE_SCRATCH_NV;
		vkGetAccelerationStructureMemoryRequirementsNV(ctx.device(), &memoryRequirementsInfo, &memoryRequirements);
		if (memoryRequirements.memoryRequirements.size > scratchSizeInBytes) scratchSizeInBytes = memoryRequirements.memoryRequirements.size;
	}

	ctx._allocate_buffer(m_blas->scratchBuffer, m_blas->scratchMem, scratchSizeInBytes, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	ctx._allocate_buffer(m_blas->resultBuffer, m_blas->resultMem, resultSizeInBytes, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	CommandBufferResource cmdBuf;
	ctx.command_buffer_create(cmdBuf, true);

	{
		VkBindAccelerationStructureMemoryInfoNV bindInfo = {};
		bindInfo.sType = VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV;
		bindInfo.accelerationStructure = m_blas->structure;
		bindInfo.memory = m_blas->resultMem;

		vkBindAccelerationStructureMemoryNV(ctx.device(), 1, &bindInfo);

		vkCmdBuildAccelerationStructureNV(cmdBuf.buf, &accelerationStructureInfo, VK_NULL_HANDLE, 0, VK_FALSE,
			m_blas->structure, VK_NULL_HANDLE, m_blas->scratchBuffer, 0);
	}

	ctx.queue_submit(cmdBuf);
	ctx.queue_wait();
	ctx.command_buffer_release(cmdBuf);
}


UnitSphere::UnitSphere(const glm::mat4x4& model, glm::vec3 color) : Geometry(model, color)
{
	m_color = color;
	m_model = model;

	static float s_aabb[6] = { -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f };

	m_aabb_buf = new BufferResource;

	Context& ctx = Context::get_context();
	ctx.buffer_create(*m_aabb_buf, sizeof(float) * 6);
	ctx.buffer_upload(*m_aabb_buf, s_aabb);

	_blas_create();
}

UnitSphere::~UnitSphere()
{
	Context& ctx = Context::get_context();
	as_release(m_blas);
	ctx.buffer_release(*m_aabb_buf);
	delete m_aabb_buf;
}

Image::Image(int width, int height, float* hdata)
{
	m_width = width;
	m_height = height;

	m_data = new BufferResource;

	Context& ctx = Context::get_context();
	ctx.buffer_create(*m_data, sizeof(float) * 4 * width * height);
	if (hdata != nullptr)
		ctx.buffer_upload(*m_data, hdata);
}

Image::~Image()
{
	Context& ctx = Context::get_context();
	ctx.buffer_release(*m_data);
	delete m_data;
}

void Image::clear()
{
	Context& ctx = Context::get_context();
	ctx.buffer_zero(*m_data);
}

void Image::to_host(void *hdata) const
{
	Context& ctx = Context::get_context();
	ctx.buffer_download(*m_data, hdata);
}


struct TriangleMeshView
{
	glm::mat3x4 normalMat;
	glm::vec4 color;
	VkDeviceAddress vertexBuf;
	VkDeviceAddress indexBuf;
};

struct SphereView
{
	glm::mat3x4 normalMat;
	glm::vec4 color;
};


struct VkGeometryInstance
{
	/// Transform matrix, containing only the top 3 rows
	float transform[12];
	/// Instance index
	uint32_t instanceId : 24;
	/// Visibility mask
	uint32_t mask : 8;
	/// Index of the hit group which will be invoked when a ray hits the instance
	uint32_t instanceOffset : 24;
	/// Instance flags, such as culling
	uint32_t flags : 8;
	/// Opaque handle of the bottom-level acceleration structure
	uint64_t accelerationStructureHandle;
};

void PathTracer::_tlas_create(const std::vector<const TriangleMesh*>& triangle_meshes, const std::vector<const UnitSphere*>& spheres)
{
	Context& ctx = Context::get_context();

	int num_triangles = (int)triangle_meshes.size();
	std::vector<VkAccelerationStructureNV> blas_triangles(num_triangles);
	std::vector<glm::mat4x4> transforms_triangles(num_triangles);
	std::vector<TriangleMeshView> tri_views(num_triangles);

	for (int i = 0; i < num_triangles; i++)
	{
		blas_triangles[i] = triangle_meshes[i]->get_blas()->structure;
		transforms_triangles[i] = triangle_meshes[i]->model();
		tri_views[i].normalMat = triangle_meshes[i]->norm();
		tri_views[i].color = { triangle_meshes[i]->color(),1.0f };
		tri_views[i].vertexBuf = ctx.buffer_get_device_address(*triangle_meshes[i]->vertex_buffer());
		tri_views[i].indexBuf = ctx.buffer_get_device_address(*triangle_meshes[i]->index_buffer());
	}

	m_triangleMeshes = new BufferResource;
	ctx.buffer_create(*m_triangleMeshes, sizeof(TriangleMeshView) * num_triangles);
	ctx.buffer_upload(*m_triangleMeshes, tri_views.data());

	int num_spheres = (int)spheres.size();
	std::vector<VkAccelerationStructureNV> blas_spheres(num_spheres);
	std::vector<glm::mat4x4> transforms_spheres(num_spheres);
	std::vector<SphereView> sphere_views(num_spheres);

	for (int i = 0; i < num_spheres; i++)
	{
		blas_spheres[i] = spheres[i]->get_blas()->structure;
		transforms_spheres[i] = spheres[i]->model();
		sphere_views[i].normalMat = spheres[i]->norm();
		sphere_views[i].color = { spheres[i]->color(),1.0f };
	}

	m_spheres = new BufferResource;
	ctx.buffer_create(*m_spheres, sizeof(SphereView) * num_spheres);
	ctx.buffer_upload(*m_spheres, sphere_views.data());

	const int num_hitgroups = 2;
	int num_instances[num_hitgroups] = { num_triangles, num_spheres };
	const VkAccelerationStructureNV* pblases[num_hitgroups] = { blas_triangles.data(), blas_spheres.data() };
	const glm::mat4x4* ptransforms[num_hitgroups] = { transforms_triangles.data(), transforms_spheres.data() };

	unsigned total = 0;
	for (int i = 0; i < num_hitgroups; i++)
		total += num_instances[i];

	VkAccelerationStructureInfoNV info = {};
	info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV;
	info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_NV;
	info.instanceCount = total;

	VkAccelerationStructureCreateInfoNV accelerationStructureInfo = {};
	accelerationStructureInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV;
	accelerationStructureInfo.info = info;

	vkCreateAccelerationStructureNV(ctx.device(), &accelerationStructureInfo, nullptr, &m_tlas->structure);

	VkDeviceSize scratchSizeInBytes, resultSizeInBytes, instanceDescsSizeInBytes;

	{

		VkAccelerationStructureMemoryRequirementsInfoNV memoryRequirementsInfo = {};
		memoryRequirementsInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_INFO_NV;
		memoryRequirementsInfo.accelerationStructure = m_tlas->structure;

		VkMemoryRequirements2 memoryRequirements;

		memoryRequirementsInfo.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_OBJECT_NV;
		vkGetAccelerationStructureMemoryRequirementsNV(ctx.device(), &memoryRequirementsInfo, &memoryRequirements);
		resultSizeInBytes = memoryRequirements.memoryRequirements.size;

		memoryRequirementsInfo.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_NV;
		vkGetAccelerationStructureMemoryRequirementsNV(ctx.device(), &memoryRequirementsInfo, &memoryRequirements);
		scratchSizeInBytes = memoryRequirements.memoryRequirements.size;

		memoryRequirementsInfo.type = VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_UPDATE_SCRATCH_NV;
		vkGetAccelerationStructureMemoryRequirementsNV(ctx.device(), &memoryRequirementsInfo, &memoryRequirements);
		if (memoryRequirements.memoryRequirements.size > scratchSizeInBytes) scratchSizeInBytes = memoryRequirements.memoryRequirements.size;

		instanceDescsSizeInBytes = total * sizeof(VkGeometryInstance);
	}

	ctx._allocate_buffer(m_tlas->scratchBuffer, m_tlas->scratchMem, scratchSizeInBytes, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	ctx._allocate_buffer(m_tlas->resultBuffer, m_tlas->resultMem, resultSizeInBytes, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	ctx._allocate_buffer(m_tlas->instancesBuffer, m_tlas->instancesMem, instanceDescsSizeInBytes, VK_BUFFER_USAGE_RAY_TRACING_BIT_NV, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

	std::vector<VkGeometryInstance> geometryInstances(total);
	unsigned k = 0;
	for (int i = 0; i < num_hitgroups; i++)
	{
		for (int j = 0; j < num_instances[i]; j++, k++)
		{
			uint64_t accelerationStructureHandle = 0;
			vkGetAccelerationStructureHandleNV(ctx.device(), pblases[i][j], sizeof(uint64_t), &accelerationStructureHandle);

			VkGeometryInstance& gInst = geometryInstances[k];
			glm::mat4x4 trans = glm::transpose(ptransforms[i][j]);
			memcpy(gInst.transform, &trans, sizeof(gInst.transform));
			gInst.instanceId = j;
			gInst.mask = 0xff;
			gInst.instanceOffset = i;
			gInst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV;
			gInst.accelerationStructureHandle = accelerationStructureHandle;
		}
	}
	
	VkDeviceSize instancesBufferSize = geometryInstances.size() * sizeof(VkGeometryInstance);
	void*        data;
	vkMapMemory(ctx.device(), m_tlas->instancesMem, 0, instancesBufferSize, 0, &data);
	memcpy(data, geometryInstances.data(), instancesBufferSize);
	vkUnmapMemory(ctx.device(), m_tlas->instancesMem);

	CommandBufferResource cmdBuf;
	ctx.command_buffer_create(cmdBuf, true);

	{
		VkBindAccelerationStructureMemoryInfoNV bindInfo = {};
		bindInfo.sType = VK_STRUCTURE_TYPE_BIND_ACCELERATION_STRUCTURE_MEMORY_INFO_NV;
		bindInfo.accelerationStructure = m_tlas->structure;
		bindInfo.memory = m_tlas->resultMem;

		vkBindAccelerationStructureMemoryNV(ctx.device(), 1, &bindInfo);

		vkCmdBuildAccelerationStructureNV(cmdBuf.buf, &info, m_tlas->instancesBuffer, 0, VK_FALSE,
			m_tlas->structure, VK_NULL_HANDLE, m_tlas->scratchBuffer, 0);
	}

	ctx.queue_submit(cmdBuf);
	ctx.queue_wait();
	ctx.command_buffer_release(cmdBuf);

}


struct ArgumentResource
{
	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorPool descriptorPool;
	VkDescriptorSet descriptorSet;
};

struct RTPipelineResource
{
	VkPipelineLayout pipelineLayout;
	VkPipeline pipeline;
	VkBuffer shaderBindingTableBuffer;
	VkDeviceMemory shaderBindingTableMem;
};

struct ComputePipelineResource
{
	VkPipelineLayout pipelineLayout;
	VkPipeline pipeline;
};

struct ImageView
{
	VkDeviceAddress data;
	int width;
	int height;
};

struct RayGenParams
{
	glm::vec4 origin;
	glm::vec4 upper_left;
	glm::vec4 ux;
	glm::vec4 uy;
	ImageView target;
	int num_iter;
};

void PathTracer::_args_create()
{
	Context& ctx = Context::get_context();

	VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[5] = { {}, {}, {}, {}, {} };
	descriptorSetLayoutBindings[0].binding = 0;
	descriptorSetLayoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV;
	descriptorSetLayoutBindings[0].descriptorCount = 1;
	descriptorSetLayoutBindings[0].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV;
	descriptorSetLayoutBindings[1].binding = 1;
	descriptorSetLayoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	descriptorSetLayoutBindings[1].descriptorCount = 1;
	descriptorSetLayoutBindings[1].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV | VK_SHADER_STAGE_COMPUTE_BIT;
	descriptorSetLayoutBindings[2].binding = 2;
	descriptorSetLayoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	descriptorSetLayoutBindings[2].descriptorCount = 1;
	descriptorSetLayoutBindings[2].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV;
	descriptorSetLayoutBindings[3].binding = 3;
	descriptorSetLayoutBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	descriptorSetLayoutBindings[3].descriptorCount = 1;
	descriptorSetLayoutBindings[3].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV;
	descriptorSetLayoutBindings[4].binding = 4;
	descriptorSetLayoutBindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	descriptorSetLayoutBindings[4].descriptorCount = 1;
	descriptorSetLayoutBindings[4].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_NV;


	VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
	descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	descriptorSetLayoutCreateInfo.bindingCount = 5;
	descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;

	vkCreateDescriptorSetLayout(ctx.device(), &descriptorSetLayoutCreateInfo, nullptr, &m_args->descriptorSetLayout);

	VkDescriptorPoolSize descriptorPoolSize[5] = { {}, {}, {}, {}, {} };
	descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV;
	descriptorPoolSize[0].descriptorCount = 1;
	descriptorPoolSize[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	descriptorPoolSize[1].descriptorCount = 1;
	descriptorPoolSize[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	descriptorPoolSize[2].descriptorCount = 1;
	descriptorPoolSize[3].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	descriptorPoolSize[3].descriptorCount = 1;
	descriptorPoolSize[4].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	descriptorPoolSize[4].descriptorCount = 1;

	VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
	descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	descriptorPoolCreateInfo.maxSets = 1;
	descriptorPoolCreateInfo.poolSizeCount = 5;
	descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;

	vkCreateDescriptorPool(ctx.device(), &descriptorPoolCreateInfo, nullptr, &m_args->descriptorPool);

	VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
	descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	descriptorSetAllocateInfo.descriptorPool = m_args->descriptorPool;
	descriptorSetAllocateInfo.descriptorSetCount = 1;
	descriptorSetAllocateInfo.pSetLayouts = &m_args->descriptorSetLayout;

	vkAllocateDescriptorSets(ctx.device(), &descriptorSetAllocateInfo, &m_args->descriptorSet);

	VkWriteDescriptorSetAccelerationStructureNV descriptorAccelerationStructureInfo = {};
	descriptorAccelerationStructureInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_NV;
	descriptorAccelerationStructureInfo.accelerationStructureCount = 1;
	descriptorAccelerationStructureInfo.pAccelerationStructures = &m_tlas->structure;

	VkDescriptorBufferInfo descriptorBufferInfo_raygen = {};
	descriptorBufferInfo_raygen.buffer = m_params_raygen->buf;
	descriptorBufferInfo_raygen.range = VK_WHOLE_SIZE;

	VkDescriptorBufferInfo descriptorBufferInfo_triangle_mesh = {};
	descriptorBufferInfo_triangle_mesh.buffer = m_triangleMeshes->buf;
	descriptorBufferInfo_triangle_mesh.range = VK_WHOLE_SIZE;

	VkDescriptorBufferInfo descriptorBufferInfo_sphere = {};
	descriptorBufferInfo_sphere.buffer = m_spheres->buf;
	descriptorBufferInfo_sphere.range = VK_WHOLE_SIZE;

	VkDescriptorBufferInfo descriptorBufferInfo_rand_states = {};
	descriptorBufferInfo_rand_states.buffer = m_rand_states->buf;
	descriptorBufferInfo_rand_states.range = VK_WHOLE_SIZE;

	std::vector<VkWriteDescriptorSet> writeDescriptorSet(3);

	writeDescriptorSet[0] = {};
	writeDescriptorSet[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writeDescriptorSet[0].dstSet = m_args->descriptorSet;
	writeDescriptorSet[0].dstBinding = 0;
	writeDescriptorSet[0].descriptorCount = 1;
	writeDescriptorSet[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV;
	writeDescriptorSet[0].pNext = &descriptorAccelerationStructureInfo;

	writeDescriptorSet[1] = {};
	writeDescriptorSet[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writeDescriptorSet[1].dstSet = m_args->descriptorSet;
	writeDescriptorSet[1].dstBinding = 1;
	writeDescriptorSet[1].descriptorCount = 1;
	writeDescriptorSet[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	writeDescriptorSet[1].pBufferInfo = &descriptorBufferInfo_raygen;

	writeDescriptorSet[2] = {};
	writeDescriptorSet[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writeDescriptorSet[2].dstSet = m_args->descriptorSet;
	writeDescriptorSet[2].dstBinding = 4;
	writeDescriptorSet[2].descriptorCount = 1;
	writeDescriptorSet[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	writeDescriptorSet[2].pBufferInfo = &descriptorBufferInfo_rand_states;

	if (m_triangleMeshes->size > 0)
	{
		VkWriteDescriptorSet write_mesh = {};
		write_mesh.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		write_mesh.dstSet = m_args->descriptorSet;
		write_mesh.dstBinding = 2;
		write_mesh.descriptorCount = 1;
		write_mesh.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		write_mesh.pBufferInfo = &descriptorBufferInfo_triangle_mesh;
		writeDescriptorSet.push_back(write_mesh);
	}

	if (m_spheres->size > 0)
	{
		VkWriteDescriptorSet write_sphere = {};
		write_sphere.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		write_sphere.dstSet = m_args->descriptorSet;
		write_sphere.dstBinding = 3;
		write_sphere.descriptorCount = 1;
		write_sphere.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		write_sphere.pBufferInfo = &descriptorBufferInfo_sphere;
		writeDescriptorSet.push_back(write_sphere);
	}	

	vkUpdateDescriptorSets(ctx.device(), (uint32_t)writeDescriptorSet.size(), writeDescriptorSet.data(), 0, nullptr);
}

void PathTracer::_args_release()
{
	Context& ctx = Context::get_context();
	vkFreeDescriptorSets(ctx.device(), m_args->descriptorPool, 1, &m_args->descriptorSet);
	vkDestroyDescriptorPool(ctx.device(), m_args->descriptorPool, nullptr);
	vkDestroyDescriptorSetLayout(ctx.device(), m_args->descriptorSetLayout, nullptr);
}

VkShaderModule _createShaderModule_from_spv(const char* fn)
{
	Context& ctx = Context::get_context();

	FILE* fp = fopen(fn, "rb");
	fseek(fp, 0, SEEK_END);
	size_t bytes = (size_t)ftell(fp);
	fseek(fp, 0, SEEK_SET);

	char* buf = new char[bytes];
	fread(buf, 1, bytes, fp);

	VkShaderModuleCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.codeSize = bytes;
	createInfo.pCode = reinterpret_cast<const uint32_t*>(buf);
	VkShaderModule shaderModule;
	vkCreateShaderModule(ctx.device(), &createInfo, nullptr, &shaderModule);

	delete[] buf;

	fclose(fp);

	return shaderModule;
}

void PathTracer::_rt_pipeline_create()
{
	Context& ctx = Context::get_context();

	VkShaderModule rayGenModule = _createShaderModule_from_spv("../shaders/raygen.spv");
	VkShaderModule missModule = _createShaderModule_from_spv("../shaders/miss.spv");
	VkShaderModule missShadowModule = _createShaderModule_from_spv("../shaders/miss_shadow.spv");
	VkShaderModule closesthit_triangles_Module = _createShaderModule_from_spv("../shaders/closesthit_triangles.spv");
	VkShaderModule intersection_spheres_Module = _createShaderModule_from_spv("../shaders/intersection_spheres.spv");
	VkShaderModule closesthit_spheres_Module = _createShaderModule_from_spv("../shaders/closesthit_spheres.spv");

	const int stage_count = 6;
	const int group_count = 5;

	VkPipelineShaderStageCreateInfo stages[stage_count] = { {}, {}, {}, {}, {}, {} };

	stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[0].stage = VK_SHADER_STAGE_RAYGEN_BIT_NV;
	stages[0].module = rayGenModule;
	stages[0].pName = "main";

	stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[1].stage = VK_SHADER_STAGE_MISS_BIT_NV;
	stages[1].module = missModule;
	stages[1].pName = "main";

	stages[2].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[2].stage = VK_SHADER_STAGE_MISS_BIT_NV;
	stages[2].module = missShadowModule;
	stages[2].pName = "main";

	stages[3].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[3].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV;
	stages[3].module = closesthit_triangles_Module;
	stages[3].pName = "main";

	stages[4].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[4].stage = VK_SHADER_STAGE_INTERSECTION_BIT_NV;
	stages[4].module = intersection_spheres_Module;
	stages[4].pName = "main";

	stages[5].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[5].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV;
	stages[5].module = closesthit_spheres_Module;
	stages[5].pName = "main";

	VkRayTracingShaderGroupCreateInfoNV groups[group_count] = { {}, {}, {}, {}, {} };
	groups[0].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV;
	groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_NV;
	groups[0].generalShader = 0;
	groups[0].closestHitShader = VK_SHADER_UNUSED_NV;
	groups[0].anyHitShader = VK_SHADER_UNUSED_NV;
	groups[0].intersectionShader = VK_SHADER_UNUSED_NV;

	groups[1].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV;
	groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_NV;
	groups[1].generalShader = 1;
	groups[1].closestHitShader = VK_SHADER_UNUSED_NV;
	groups[1].anyHitShader = VK_SHADER_UNUSED_NV;
	groups[1].intersectionShader = VK_SHADER_UNUSED_NV;

	groups[2].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV;
	groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_NV;
	groups[2].generalShader = 2;
	groups[2].closestHitShader = VK_SHADER_UNUSED_NV;
	groups[2].anyHitShader = VK_SHADER_UNUSED_NV;
	groups[2].intersectionShader = VK_SHADER_UNUSED_NV;

	groups[3].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV;
	groups[3].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_NV;
	groups[3].generalShader = VK_SHADER_UNUSED_NV;
	groups[3].closestHitShader = 3;
	groups[3].anyHitShader = VK_SHADER_UNUSED_NV;
	groups[3].intersectionShader = VK_SHADER_UNUSED_NV;

	groups[4].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV;
	groups[4].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_NV;
	groups[4].generalShader = VK_SHADER_UNUSED_NV;
	groups[4].closestHitShader = 5;
	groups[4].anyHitShader = VK_SHADER_UNUSED_NV;
	groups[4].intersectionShader = 4;

	VkDescriptorSetLayout descriptorSetLayouts[1] = { m_args->descriptorSetLayout };

	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
	pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutCreateInfo.setLayoutCount = 1;
	pipelineLayoutCreateInfo.pSetLayouts = descriptorSetLayouts;

	vkCreatePipelineLayout(ctx.device(), &pipelineLayoutCreateInfo, nullptr, &m_rt_pipeline->pipelineLayout);

	VkRayTracingPipelineCreateInfoNV rayPipelineInfo = {};
	rayPipelineInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_NV;
	rayPipelineInfo.stageCount = stage_count;
	rayPipelineInfo.pStages = stages;
	rayPipelineInfo.groupCount = group_count;
	rayPipelineInfo.pGroups = groups;
	rayPipelineInfo.maxRecursionDepth = 1;
	rayPipelineInfo.layout = m_rt_pipeline->pipelineLayout;

	vkCreateRayTracingPipelinesNV(ctx.device(), nullptr, 1, &rayPipelineInfo, nullptr, &m_rt_pipeline->pipeline);

	vkDestroyShaderModule(ctx.device(), closesthit_spheres_Module, nullptr);
	vkDestroyShaderModule(ctx.device(), intersection_spheres_Module, nullptr);
	vkDestroyShaderModule(ctx.device(), closesthit_triangles_Module, nullptr);
	vkDestroyShaderModule(ctx.device(), missShadowModule, nullptr);
	vkDestroyShaderModule(ctx.device(), missModule, nullptr);
	vkDestroyShaderModule(ctx.device(), rayGenModule, nullptr);

	// shader binding table
	unsigned progIdSize = ctx.raytracing_properties().shaderGroupHandleSize;
	unsigned sbtSize = progIdSize * group_count;

	ctx._allocate_buffer(m_rt_pipeline->shaderBindingTableBuffer, m_rt_pipeline->shaderBindingTableMem, sbtSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

	unsigned char* shaderHandleStorage = (unsigned char*)malloc(group_count *progIdSize);
	vkGetRayTracingShaderGroupHandlesNV(ctx.device(), m_rt_pipeline->pipeline, 0, group_count, progIdSize * group_count, shaderHandleStorage);

	unsigned char* vData;
	vkMapMemory(ctx.device(), m_rt_pipeline->shaderBindingTableMem, 0, sbtSize, 0, (void**)&vData);
	memcpy(vData, shaderHandleStorage, progIdSize * group_count);
	vkUnmapMemory(ctx.device(), m_rt_pipeline->shaderBindingTableMem);

	free(shaderHandleStorage);

}


void PathTracer::_rt_pipeline_release()
{
	Context& ctx = Context::get_context();
	ctx._release_buffer(m_rt_pipeline->shaderBindingTableBuffer, m_rt_pipeline->shaderBindingTableMem);
	vkDestroyPipelineLayout(ctx.device(), m_rt_pipeline->pipelineLayout, nullptr);
	vkDestroyPipeline(ctx.device(), m_rt_pipeline->pipeline, nullptr);
}

void PathTracer::_comp_pipeline_create()
{
	Context& ctx = Context::get_context();

	VkShaderModule finalModule = _createShaderModule_from_spv("../shaders/final.spv");

	VkPipelineShaderStageCreateInfo computeShaderStageInfo = {};
	computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	computeShaderStageInfo.module = finalModule;
	computeShaderStageInfo.pName = "main";

	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
	pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutCreateInfo.setLayoutCount = 1;
	pipelineLayoutCreateInfo.pSetLayouts = &m_args->descriptorSetLayout;

	vkCreatePipelineLayout(ctx.device(), &pipelineLayoutCreateInfo, 0, &m_comp_pipeline->pipelineLayout);

	VkComputePipelineCreateInfo pipelineInfo = {};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pipelineInfo.stage = computeShaderStageInfo;
	pipelineInfo.layout = m_comp_pipeline->pipelineLayout;

	vkCreateComputePipelines(ctx.device(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_comp_pipeline->pipeline);

	vkDestroyShaderModule(ctx.device(), finalModule, nullptr);
}

void PathTracer::_comp_pipeline_release()
{
	Context& ctx = Context::get_context();
	vkDestroyPipelineLayout(ctx.device(), m_comp_pipeline->pipelineLayout, nullptr);
	vkDestroyPipeline(ctx.device(), m_comp_pipeline->pipeline, nullptr);
}

#include "rand_state_init.hpp"

PathTracer::PathTracer(Image* target, const std::vector<const TriangleMesh*>& triangle_meshes, const std::vector<const UnitSphere*>& spheres)
{
	Context& ctx = Context::get_context();

	m_target = target;

	m_tlas = new AccelerationResource;
	_tlas_create(triangle_meshes, spheres);

	m_params_raygen = new BufferResource;
	ctx.buffer_create(*m_params_raygen, sizeof(RayGenParams));

	m_rand_states = new BufferResource;
	ctx.buffer_create(*m_rand_states, sizeof(unsigned) * 6 * m_target->width()*m_target->height());
	
	m_args = new ArgumentResource;
	m_rt_pipeline = new RTPipelineResource;
	m_comp_pipeline = new ComputePipelineResource;

	_args_create();
	_rt_pipeline_create();
	_comp_pipeline_create();

	set_camera({ 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, -1.0f }, { 0.0f, 1.0f, 0.0f }, 90.0f);

	m_cmdbuf = new CommandBufferResource;
	ctx.command_buffer_create(*m_cmdbuf);

	_rand_init_cuda();
}

PathTracer::~PathTracer()
{
	Context& ctx = Context::get_context();

	ctx.command_buffer_release(*m_cmdbuf);
	delete m_cmdbuf;

	_comp_pipeline_release();
	delete m_comp_pipeline;

	_rt_pipeline_release();
	delete m_rt_pipeline;

	_args_release();
	delete m_args;

	ctx.buffer_release(*m_rand_states);	
	delete m_rand_states;

	ctx.buffer_release(*m_params_raygen);
	delete m_params_raygen;

	as_release(m_tlas);
	delete m_tlas;

	ctx.buffer_release(*m_spheres);
	ctx.buffer_release(*m_triangleMeshes);

	delete m_spheres;
	delete m_triangleMeshes;
}

void PathTracer::_update_args(int num_iter)
{
	Context& ctx = Context::get_context();

	RayGenParams raygen_params;
	ImageView target_view;
	{
		target_view.data = ctx.buffer_get_device_address(*m_target->data());
		target_view.width = m_target->width();
		target_view.height = m_target->height();
	}
	raygen_params.target = target_view;
	raygen_params.origin = glm::vec4(m_origin, 1.0f);
	raygen_params.upper_left = glm::vec4(m_upper_left, 1.0f);
	raygen_params.ux = glm::vec4(m_ux, 1.0f);
	raygen_params.uy = glm::vec4(m_uy, 1.0f);
	raygen_params.num_iter = num_iter;

	ctx.buffer_upload(*m_params_raygen, &raygen_params);
}

void PathTracer::set_camera(glm::vec3 lookfrom, glm::vec3 lookat, glm::vec3 vup, float vfov)
{
	Context& ctx = Context::get_context();

	float focus_dist = 1.0f;

	m_origin = lookfrom;

	float theta = vfov * PI / 180.0f;
	float half_height = tanf(theta*0.5f)*focus_dist;
	float size_pix = half_height * 2.0f / (float)m_target->height();
	float half_width = size_pix * (float)m_target->width()*0.5f;

	glm::vec3 axis_z = normalize(lookfrom - lookat);
	glm::vec3 axis_x = normalize(cross(vup, axis_z));
	glm::vec3 axis_y = cross(axis_z, axis_x);

	glm::vec3 plane_center = lookfrom - axis_z * focus_dist;
	m_upper_left = plane_center - axis_x * half_width + axis_y * half_height;
	m_ux = size_pix * axis_x;
	m_uy = -size_pix * axis_y;
}

void PathTracer::trace(int num_iter)
{
	_update_args(num_iter);
	Context& ctx = Context::get_context();

	m_target->clear();

	unsigned progIdSize = ctx.raytracing_properties().shaderGroupHandleSize;

	vkCmdBindPipeline(m_cmdbuf->buf, VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, m_rt_pipeline->pipeline);
	vkCmdBindDescriptorSets(m_cmdbuf->buf, VK_PIPELINE_BIND_POINT_RAY_TRACING_NV, m_rt_pipeline->pipelineLayout, 0, 1, &m_args->descriptorSet, 0, nullptr);

	for (int i = 0; i < num_iter; i++)
	{
		vkCmdTraceRaysNV(m_cmdbuf->buf,
			m_rt_pipeline->shaderBindingTableBuffer, 0,
			m_rt_pipeline->shaderBindingTableBuffer, progIdSize, progIdSize,
			m_rt_pipeline->shaderBindingTableBuffer, progIdSize * 3, progIdSize,
			VK_NULL_HANDLE, 0, 0, m_target->width(), m_target->height(), 1);

		VkMemoryBarrier memoryBarrier = {};
		memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
 
		vkCmdPipelineBarrier(m_cmdbuf->buf, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_NV, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_NV | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
			
	}

	int group_x = (m_target->width() + 15) / 16;
	int group_y = (m_target->height() + 15) / 16;
	{
		vkCmdBindPipeline(m_cmdbuf->buf, VK_PIPELINE_BIND_POINT_COMPUTE, m_comp_pipeline->pipeline);
		vkCmdBindDescriptorSets(m_cmdbuf->buf, VK_PIPELINE_BIND_POINT_COMPUTE, m_comp_pipeline->pipelineLayout, 0, 1, &m_args->descriptorSet, 0, nullptr);
		vkCmdDispatch(m_cmdbuf->buf, group_x, group_y, 1);
	}

	ctx.queue_submit(*m_cmdbuf);
	ctx.queue_wait();
}


