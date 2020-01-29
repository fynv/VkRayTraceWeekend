#pragma once

#include <glm.hpp>
#include <vector>

struct AccelerationResource;
struct BufferResource;

class Geometry
{
public:
	const glm::vec3& color() const { return m_color; }
	const glm::mat4x4& model() const { return m_model; }
	const glm::mat4x4& norm() const { return m_norm_mat; }
	AccelerationResource* get_blas() const { return m_blas; }

	Geometry(const glm::mat4x4& model, glm::vec3 color);
	virtual ~Geometry();

protected:
	glm::vec3 m_color;
	glm::mat4x4 m_model;
	glm::mat4x4 m_norm_mat;
	AccelerationResource* m_blas;

};


struct Vertex
{
	glm::vec3 Position;
	glm::vec3 Normal;
	glm::vec2 TexCoord;
};

class TriangleMesh : public Geometry
{
public:
	BufferResource* vertex_buffer() const { return m_vertexBuffer; }
	BufferResource* index_buffer() const { return m_indexBuffer; }
	
	TriangleMesh(const glm::mat4x4& model, const std::vector<Vertex>& vertices, const std::vector<unsigned>& indices, glm::vec3 color = { 1.0f, 1.0f, 1.0f });
	virtual ~TriangleMesh();

private:
	void _blas_create();

	unsigned m_vertexCount;
	unsigned m_indexCount;

	BufferResource* m_vertexBuffer;
	BufferResource* m_indexBuffer;
};


class UnitSphere : public Geometry
{
public:
	BufferResource* aabb_buffer() const { return m_aabb_buf; }

	UnitSphere(const glm::mat4x4& model, glm::vec3 color = { 1.0f, 1.0f, 1.0f });
	virtual ~UnitSphere();


private:
	void _blas_create();
	BufferResource* m_aabb_buf;

};

class Image
{
public:
	BufferResource* data() const { return m_data; }
	int width() const { return m_width; }
	int height() const { return m_height; }

	Image(int width, int height, float* hdata = nullptr);
	~Image();

	void clear();
	void to_host(void *hdata) const;

private:
	BufferResource* m_data;
	int m_width;
	int m_height;

};

struct ArgumentResource;
struct RTPipelineResource;
struct ComputePipelineResource;
struct CommandBufferResource;

class PathTracer
{
public:
	PathTracer(Image* target, const std::vector<const TriangleMesh*>& triangle_meshes, const std::vector<const UnitSphere*>& spheres);
	~PathTracer();

	void set_camera(glm::vec3 lookfrom, glm::vec3 lookat, glm::vec3 vup, float vfov);
	void trace(int num_iter = 100);

private:
	void _update_args(int num_iter);

	void _tlas_create(const std::vector<const TriangleMesh*>& triangle_meshes, const std::vector<const UnitSphere*>& spheres);
	void _args_create();
	void _args_release();
	void _rt_pipeline_create();
	void _rt_pipeline_release();

	void _comp_pipeline_create();
	void _comp_pipeline_release();

	void _rand_init_cpu();
	void _rand_init_cuda();

	AccelerationResource* m_tlas;
	Image* m_target;
	BufferResource* m_triangleMeshes;
	BufferResource* m_spheres;
	glm::vec3 m_origin;
	glm::vec3 m_upper_left;
	glm::vec3 m_ux;
	glm::vec3 m_uy;

	BufferResource* m_params_raygen;
	BufferResource* m_rand_states;
	
	ArgumentResource* m_args;
	RTPipelineResource* m_rt_pipeline;
	ComputePipelineResource* m_comp_pipeline;
	CommandBufferResource* m_cmdbuf;
	
};

