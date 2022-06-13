//=============================================================================================
// Mintaprogram: Zold haromszog. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Meszaros Peter
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

const char* vertexSource = R"(
	#version 330
    precision highp float;

	uniform vec3 wLookAt, wRight, wUp;

	layout(location = 0) in vec2 cCamWindowVertex;
	out vec3 p;

	void main() {
		gl_Position = vec4(cCamWindowVertex, 0, 1);
		p = wLookAt + wRight * cCamWindowVertex.x + wUp * cCamWindowVertex.y;
	}
)";

const char* fragmentSource = R"(
	#version 330
    precision highp float;
	#define M_PI 3.14159265359

	struct Material {
		vec3 ka, kd, ks;
		float  shininess;
		vec3 F0;
		int type;
	};

	struct Light {
		vec3 position;
		vec3 Le, La;
	};

	struct ImplicitSurface{
		float radius, a, b, c;
		vec3 center;
	};

	struct Hit {
		float t;
		vec3 position, normal;
		int material;
		int faceNumber;
	};

	struct Ray {
		vec3 start, dir;
	};

	const int nObjFaces = 12;
	const int nObjFaceVertices = 5;
	const float distFromSide = 0.1f*sin(54/180.0f*M_PI);

	struct Face {
		vec3 vertices[nObjFaceVertices];
		vec3 sideVectors[nObjFaceVertices];
		vec3 normal;
		mat4 rotationStart[2];
		mat4 rotationDir[2];
	};

	uniform vec3 wEye; 
	uniform Light light;     
	uniform Material materials[3];
	uniform Face faces[nObjFaces];
	uniform ImplicitSurface implicitSurface;

	in  vec3 p;
	out vec4 fragmentColor;

	vec3 normalImplicitSurface(const ImplicitSurface surface, const vec3 point){
		return normalize(vec3(2.0f*surface.a*point.x, 2.0f*surface.b*point.y, -surface.c));
	}

	Hit intersectImplicitSurface(const ImplicitSurface surface, const Ray ray){
		Hit hit;
		hit.t = -1;
		float A = surface.a * ray.dir.x * ray.dir.x + surface.b * ray.dir.y * ray.dir.y;
		float B = 2.0f * surface.a * ray.start.x * ray.dir.x + 2.0f * surface.b * ray.start.y * ray.dir.y - surface.c * ray.dir.z;
		float C = surface.a * ray.start.x * ray.start.x + surface.b * ray.start.y * ray.start.y -surface.c * ray.start.z;	
		float discr = B * B - 4.0f * A * C;
		if (discr < 0) return hit;	
		float sqrt_discr = sqrt(discr);
		float t1 = (-B + sqrt_discr) / 2.0f / A;
		float t2 = (-B - sqrt_discr) / 2.0f / A;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = normalImplicitSurface(surface, hit.position);
		hit.material = 1;
		if (length(hit.position - surface.center) > surface.radius) {
			vec3 dist = ray.start - surface.center;
			float a = dot(ray.dir, ray.dir);
			float b = dot(dist, ray.dir) * 2.0f;
			float c = dot(dist, dist) - surface.radius * surface.radius;
			discr = b * b - 4.0f * a * c;
			if (discr < 0){
				hit.t = -1;
				return hit;
			}
			sqrt_discr = sqrt(discr);
			float t1_sphere = (-b + sqrt_discr) / 2.0f / a;
			float t2_sphere = (-b - sqrt_discr) / 2.0f / a;
			if (t1_sphere <= 0){
				hit.t = -1;
				return hit;
			}
			if (!(t2_sphere < hit.t && hit.t < t1_sphere)) hit.t = t2_sphere;
		}
		return hit;
	}

	Hit intersectFace(const Ray ray, int face){
		Hit hit;
		hit.t = -1;
		vec3 normal = faces[face].normal;
		float t = dot(faces[face].vertices[0] - ray.start, normal) / dot(ray.dir, normal);
		if (t <= 0) return hit;
		hit.t = t;
		hit.normal = normal;
		hit.material = 2;
		hit.faceNumber = face;
		hit.position = ray.start + ray.dir*t;
		for (int i = 0; i < nObjFaceVertices; i++)
			if (dot(cross(faces[face].sideVectors[i], (hit.position - faces[face].vertices[i])), normal) <= 0)
				hit.t = -1;
		for (int i = 0; i < nObjFaceVertices; i++){
			vec3 normalizedSideVector = normalize(faces[face].sideVectors[i]);
			vec3 p = hit.position - faces[face].vertices[i];
			float d = length(p - dot(normalizedSideVector, p) * normalizedSideVector);
			if (d < distFromSide) {
				hit.material = 0;
				return hit;
			}
		}
		return hit;
	}

	Hit intersectPolyhedron(const Ray ray){
		Hit hit;
		hit.t = -1;
		for (int face = 0; face < nObjFaces; face++){
			Hit faceHit = intersectFace(ray, face);
			if (faceHit.t > 0 && (faceHit.t < hit.t || hit.t == -1))
				hit = faceHit;
		}
		return hit;
	}

	Hit firstIntersect(const Ray ray) {
		Hit bestHit;
		bestHit.t = -1;
		Hit hit = intersectImplicitSurface(implicitSurface, ray);
		if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))
			bestHit = hit;
		hit = intersectPolyhedron(ray);
		if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))
			bestHit = hit;
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	vec3 Fresnel(vec3 F0, float cosTheta) { 
		return F0 + (vec3(1, 1, 1) - F0) * pow(cosTheta, 5);
	}

	const float epsilon = 0.0001f;
	const int maxdepth = 5;

	vec3 trace(Ray ray) {
		vec3 weight = vec3(1, 1, 1);
		vec3 outRadiance = vec3(0,0,0);
		int portal = 0;
		int parity = 1;
		while (portal <= maxdepth) {
			Hit hit = firstIntersect(ray);
			if (hit.t < 0) return weight * light.La;
			if (materials[hit.material].type == 0) {
				vec3 lDir = light.position-hit.position;
				float magnitude = dot(lDir, lDir);
				lDir = normalize(lDir);
				outRadiance += weight * materials[hit.material].ka * light.La;
				float cosTheta = dot(hit.normal, lDir);
				if (cosTheta > 0) {
					outRadiance += weight * light.Le/magnitude * materials[hit.material].kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + lDir);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance += weight * light.Le/magnitude * materials[hit.material].ks * pow(cosDelta, materials[hit.material].shininess);
				}
				return outRadiance+weight*light.La*materials[hit.material].ka;
			}else{
				if (materials[hit.material].type == 1){
					weight *= Fresnel(materials[hit.material].F0, dot(-ray.dir, hit.normal));
					ray.dir = reflect(ray.dir, hit.normal);
					ray.start = hit.position + hit.normal * epsilon;
				}else{
					portal++;
					parity = 3-parity;
					ray.dir = (vec4(reflect(ray.dir, hit.normal), 1)*faces[hit.faceNumber].rotationDir[parity-1]).xyz;
					ray.start = (vec4(hit.position, 1)*faces[hit.faceNumber].rotationStart[parity-1]).xyz + hit.normal*epsilon;
				}
			}
		}
		return weight*light.La;
	}

	void main() {
		Ray ray;
		ray.start = wEye; 
		ray.dir = normalize(p - wEye);
		fragmentColor = vec4(trace(ray), 1); 
	}
)";

enum MaterialType {ROUGH = 0, REFLECTIVE = 1, PORTAL = 2};

class Material {
public:
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	MaterialType type;

	static Material* createRoughMaterial(vec3 _kd, vec3 _ks, float _shininess){
		Material* m = new Material();
		m->ka = _kd;
		m->kd = _kd;
		m->ks = _ks;
		m->shininess = _shininess;
		m->type = ROUGH;
		return m;
	}

	static Material* createSmoothMaterial(vec3 n, vec3 kappa) {
		Material* m = new Material();
		vec3 numerator = (n - vec3(1, 1, 1)) * (n - vec3(1, 1, 1)) + kappa * kappa;
		vec3 denominator = (n + vec3(1, 1, 1)) * (n + vec3(1, 1, 1)) + kappa * kappa;
		m->F0 = vec3(numerator.x / denominator.x, numerator.y / denominator.y, numerator.z / denominator.z);
		m->type = REFLECTIVE;
		return m;
	}

	static Material* createPortalMaterial() {
		Material* m = new Material();
		m->type = PORTAL;
		return m;
	}
};

struct Face {
	std::vector<vec3> vertices;
	std::vector<vec3> sideVectors;
	vec3 normal;
	vec3 center;
	Face(std::vector<vec3>& v) {
		vertices.push_back(v[0]);
		for (int i = 1; i < v.size(); i++) {
			vertices.push_back(v[i]);
			sideVectors.push_back(v[i] - v[i - 1]);
		}
		sideVectors.push_back(v[0] - v[v.size() - 1]);
		normal = normalize(cross(vertices[1] - vertices[0], vertices[2] - vertices[0]));
		center = vec3(0, 0, 0);
		for (vec3 v : vertices)
			center = center + v;
		center = center * 1.0f / vertices.size();
	}
	mat4 rotateAroundCenter(float angle) {
		return TranslateMatrix(-center)*RotationMatrix(angle, normal)* TranslateMatrix(center);
	}
	mat4 rotateAroundNormal(float angle) {
		return RotationMatrix(angle, normal);
	}
};

struct Polyhedron {
	std::vector<vec3> vertices;
	std::vector<Face*> faces;

	void createDodecahedron() {
		vertices = std::vector<vec3>{ vec3(0, 0.618, 1.618), vec3(0, -0.618, 1.618), vec3(0, -0.618, -1.618),vec3(0, 0.618, -1.618),vec3(1.618, 0, 0.618),vec3(-1.618, 0, 0.618),vec3(-1.618, 0, -0.618)
										,vec3(1.618, 0, -0.618),vec3(0.618, 1.618, 0),vec3(-0.618, 1.618, 0),vec3(-0.618, -1.618, 0),vec3(0.618, -1.618, 0), vec3(1, 1, 1),vec3(-1, 1, 1),vec3(-1, -1, 1)
										,vec3(1, -1, 1), vec3(1, -1, -1), vec3(1, 1, -1), vec3(-1, 1, -1), vec3(-1, -1, -1) };
		int sides[] = { 1, 2, 16, 5, 13, 1, 13, 9, 10, 14, 1, 14, 6, 15, 2, 2, 15, 11, 12, 16, 3, 4, 18, 8, 17, 3, 17, 12, 11, 20, 3, 20, 7, 19, 4, 19, 10, 9, 18, 4, 16,
						12, 17, 8, 5, 5, 8, 18, 9, 13, 14, 10, 19, 7, 6, 6, 7, 20, 11, 15 };
		for (int i = 0; i < 12; i++) {
			std::vector<vec3> v;
			for (int j = 0; j < 5; j++)
				v.push_back(vertices[sides[i * 5 + j] - 1]);
			faces.push_back(new Face(v));
		}
	}
};



struct Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye;
		lookat = _lookat;
		fov = _fov;
		vec3 w = eye - lookat;
		float f = length(w);
		right = normalize(cross(vup, w)) * f * tanf(fov / 2);
		up = normalize(cross(w, right)) * f * tanf(fov / 2);
	}
	void Animate(float dt) {
		eye = vec3((eye.x - lookat.x) * cos(dt) + (eye.y - lookat.y) * sin(dt) + lookat.x,
			-(eye.x - lookat.x) * sin(dt) + (eye.y - lookat.y) * cos(dt) + lookat.y,
			eye.z);
		set(eye, lookat, up, fov);
	}
};

struct Light {
	vec3 position;
	vec3 Le, La;
	Light(vec3 _position, vec3 _Le, vec3 _La) :position(_position), Le(_Le), La(_La) {}
};

struct ImplicitSurface {
	float radius, a, b, c;
	vec3 center;
	ImplicitSurface(vec3 _center, float _radius, float _a, float _b, float _c)
		:center(_center), radius(_radius), a(_a), b(_b), c(_c) {}
};

//mintaprogram
class Shader : public GPUProgram {
public:
	void setUniformMaterials(const std::vector<Material*>& materials) {
		char name[256];
		for (unsigned int mat = 0; mat < materials.size(); mat++) {
			sprintf(name, "materials[%d].ka", mat); setUniform(materials[mat]->ka, name);
			sprintf(name, "materials[%d].kd", mat); setUniform(materials[mat]->kd, name);
			sprintf(name, "materials[%d].ks", mat); setUniform(materials[mat]->ks, name);
			sprintf(name, "materials[%d].shininess", mat); setUniform(materials[mat]->shininess, name);
			sprintf(name, "materials[%d].F0", mat); setUniform(materials[mat]->F0, name);
			sprintf(name, "materials[%d].type", mat); setUniform(materials[mat]->type, name);
		}
	}

	void setUniformLight(const Light* light) {
		setUniform(light->La, "light.La");
		setUniform(light->Le, "light.Le");
		setUniform(light->position, "light.position");
	}

	void setUniformCamera(const Camera& camera) {
		setUniform(camera.eye, "wEye");
		setUniform(camera.lookat, "wLookAt");
		setUniform(camera.right, "wRight");
		setUniform(camera.up, "wUp");
	}

	void setUniformImplicitSurface(const ImplicitSurface* surface) {
		setUniform(surface->center, "implicitSurface.center");
		setUniform(surface->radius, "implicitSurface.radius");
		setUniform(surface->a, "implicitSurface.a");
		setUniform(surface->b, "implicitSurface.b");
		setUniform(surface->c, "implicitSurface.c");
	}

	void setUniformPolyhedron(const Polyhedron* poly) {
		char name[256];
		for (unsigned int i = 0; i < poly->faces.size(); i++) {
			sprintf(name, "faces[%d].rotationStart[0]", i);
			setUniform(poly->faces[i]->rotateAroundCenter(72 / 180.0f * M_PI), name);
			sprintf(name, "faces[%d].rotationStart[1]", i);
			setUniform(poly->faces[i]->rotateAroundCenter(-72 / 180.0f * M_PI), name);
			sprintf(name, "faces[%d].rotationDir[0]", i);
			setUniform(poly->faces[i]->rotateAroundNormal(72 / 180.0f * M_PI), name);
			sprintf(name, "faces[%d].rotationDir[1]", i);
			setUniform(poly->faces[i]->rotateAroundNormal(-72 / 180.0f * M_PI), name);
			sprintf(name, "faces[%d].normal", i);
			setUniform(poly->faces[i]->normal, name);
			for (int j = 0; j < poly->faces[i]->vertices.size(); j++) {
				sprintf(name, "faces[%d].vertices[%d]", i, j);
				setUniform(poly->faces[i]->vertices[j], name);
				sprintf(name, "faces[%d].sideVectors[%d]", i, j);
				setUniform(poly->faces[i]->sideVectors[j], name);
			}
		}
	}
};

class Scene {
public:
	Light* light;
	Camera camera;
	std::vector<Material*> materials;
	Polyhedron* dodeca;
	ImplicitSurface* implicitSurface;
public:
	void build() {
		vec3 eye = vec3(-0.8f, 0.8f, 0.5f);
		vec3 vup = vec3(0, 0, 1);
		vec3 lookat = vec3(0, 0, 0);
		float fov = 45 * (float)M_PI / 180;
		camera.set(eye, lookat, vup, fov);
		light = new Light(vec3(-0.3f, 0.5f, 0.4f), vec3(2.0f, 2.0f, 2.0f), vec3(0.38f, 0.55f, 0.55f));

		materials.push_back(Material::createRoughMaterial(vec3(0.75f, 0.46f, 0.37f)*0.8f, vec3(1, 1, 1), 50));
		materials.push_back(Material::createSmoothMaterial(vec3(0.17f, 0.35f, 1.5f), vec3(3.0f, 2.7f, 1.9f)));
		materials.push_back(Material::createPortalMaterial());

		dodeca = new Polyhedron();
		dodeca->createDodecahedron();
		implicitSurface = new ImplicitSurface(vec3(0, 0, 0), 0.3f, 0.25f, 0.25f, 0.3f);
	}

	void setUniform(Shader& shader) {
		shader.setUniformMaterials(materials);
		shader.setUniformLight(light);
		shader.setUniformCamera(camera);
		shader.setUniformPolyhedron(dodeca);
		shader.setUniformImplicitSurface(implicitSurface);
	}

	void Animate(float dt) { camera.Animate(dt); }
};

Shader shader;
Scene scene;

//eloadasdia/mintaprogram
class FullScreenTexturedQuad {
	unsigned int vao = 0;
public:
	void create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);	

		unsigned int vbo;	
		glGenBuffers(1, &vbo);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	}

	void Draw() {
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};

FullScreenTexturedQuad fullScreenTexturedQuad;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	fullScreenTexturedQuad.create();

	shader.create(vertexSource, fragmentSource, "fragmentColor");
	shader.Use();
	scene.setUniform(shader);
}

void onDisplay() {
	glClearColor(1.0f, 0.5f, 0.8f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	shader.setUniformCamera(scene.camera);
	fullScreenTexturedQuad.Draw();

	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
}

void onMouse(int button, int state, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
}

void onIdle() {
	scene.Animate(0.01f);
	glutPostRedisplay();
}
