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

//eloadasdia
template<class T> struct Dnum {
	float f;
	T d;
	Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
	Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
	Dnum operator*(Dnum r) {
		return Dnum(f * r.f, f * r.d + d * r.f);
	}
	Dnum operator/(Dnum r) {
		return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
	}
};

template<class T> Dnum<T> Sin(Dnum<T> g) { return  Dnum<T>(sinf(g.f), cosf(g.f) * g.d); }
template<class T> Dnum<T> Cos(Dnum<T>  g) { return  Dnum<T>(cosf(g.f), -sinf(g.f) * g.d); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) {
	return  Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d);
}

typedef Dnum<vec2> Dnum2;

vec4 qmul(vec4 q1, vec4 q2) {
	vec3 d1(q1.x, q1.y, q1.z), d2(q2.x, q2.y, q2.z);
	vec3 temp = d2 * q1.w + d1 * q2.w + cross(d1, d2);
	return vec4(temp.x, temp.y, temp.z,  q1.w * q2.w -dot(d1, d2));
}

const int tessellationLevel = 100;

//eloadasdia
struct Camera {
	vec3 wEye, wLookat, wVup;
	float fov, asp, fp, bp;
public:
	Camera() {
		asp = (float)windowWidth / windowHeight;
		fov = 75.0f * (float)M_PI / 180.0f;
		fp = 1; bp = 20;
	}
	mat4 V() {
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);
	}

	virtual mat4 P() {
		return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
			0, 1 / tan(fov / 2), 0, 0,
			0, 0, -(fp + bp) / (bp - fp), -1,
			0, 0, -2 * fp * bp / (bp - fp), 0);
	}
};

struct Material {
	vec3 kd, ks, ka;
	float shininess;
};

struct Light {
	vec3 La, Le;
	vec4 wLightPos;
	vec4 startPos;
	vec4 pivot;
	Light() {}
	Light(vec3 _La, vec3 _Le, vec4 _wLightPos, vec4 _pivot) : La(_La), Le(_Le), wLightPos(_wLightPos), pivot(_pivot) {
		startPos = wLightPos;
	}
	void Animate(float tstart, float tend) {
		float t = tend;
		vec4 q(cos(t / 4), sin(t / 4) * cos(t) / 2, sin(t / 4) * sin(t) / 2, sin(t / 4) * sqrtf(3 / 4));
		vec4 qinv = vec4(-q.x, -q.y, -q.z, q.w);
		wLightPos = pivot;
		wLightPos = startPos - pivot;
		wLightPos = qmul(qmul(q, wLightPos), qinv);
		wLightPos = wLightPos + pivot;
	}
};

struct RenderState {
	mat4	           MVP, M, Minv, V, P;
	Material* material;
	Light *lights;
	vec3	           wEye;
};

class PhongShader : public GPUProgram {
protected:

	//eloadasdia/peldakod
	char * vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv;
		uniform Light[8] lights;
		uniform int nLights;
		uniform vec3  wEye;

		layout(location = 0) in vec3  vtxPos;  
		layout(location = 1) in vec3  vtxNorm; 

		out vec3 wNormal;
		out vec3 wView;    
		out vec3 wLight[8];	
		out float z;

		void main() {
			gl_Position =  vec4(vtxPos, 1) * MVP;
			vec4 wPos = vec4(vtxPos, 1) * M;
			z = wPos.z;
			for(int i = 0; i < nLights; i++)
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		}
	)";

	char * fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;  
		uniform int   nLights;

		in float z;
		in  vec3 wNormal;    
		in  vec3 wView;   
		in  vec3 wLight[8]; 
		
        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;
			vec3 ka = material.ka;
			vec3 kd = material.kd;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				float length = sqrt(dot(wLight[i]-wView, wLight[i]-wView));
				radiance += ka * lights[i].La + (kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le/dot(wLight[i]-wView, wLight[i]-wView);

			}
			fragmentColor = vec4(radiance, 1);
		}
	)";

public:
	PhongShader() {
		create(vertexSource, fragmentSource, "fragmentColor");
	}

	//eloadasdia/peldakod
	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shininess, name + ".shininess");
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniformMaterial(*state.material, "material");

		setUniform((int)2, "nLights");
		for (unsigned int i = 0; i < 2; i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};

class SheetPhongShader : public PhongShader {
protected:
	char* fragmentSource2 = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;  
		uniform int   nLights;

		in float z;
		in  vec3 wNormal; 
		in  vec3 wView;     
		in  vec3 wLight[8];   
		
        out vec4 fragmentColor;

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;
			vec3 ka = material.ka;
			vec3 kd = material.kd;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				if (z != 0)
					kd = kd + kd*floor(z*10.0f)/10.0f;
				radiance += ka * lights[i].La + (kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le/dot(wLight[i]-wView, wLight[i]-wView);

			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
		
public:
	SheetPhongShader() {
		create(vertexSource, fragmentSource2, "fragmentColor");
	}
};

class Geometry {
protected:
	unsigned int vao, vbo; 
public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	virtual void Draw() = 0;
	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

struct VertexData {
	vec3 position, normal;
	vec2 texcoord;
};

//eloadasdia/peldakod
class ParamSurface : public Geometry {

	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() { nVtxPerStrip = nStrips = 0; }

	virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;

	VertexData GenVertexData(float u, float v) {
		VertexData vtxData;
		vtxData.texcoord = vec2(u, v);
		Dnum2 X, Y, Z;
		Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
		eval(U, V, X, Y, Z);
		vtxData.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
		vtxData.normal = cross(drdU, drdV);
		return vtxData;
	}

	void create(int N = tessellationLevel, int M = tessellationLevel) {
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;	// vertices on the CPU
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1); 
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}

	void Draw() {
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++)
			glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
	}
};


class Sphere : public ParamSurface {
public:
	Sphere() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * (float)M_PI, V = V * (float)M_PI;
		X = Cos(U) * Sin(V); Y = Sin(U) * Sin(V); Z = Cos(V);
	}
};



class SheetGeometry : public ParamSurface {
	struct Weight {
		vec2 pos;
		float mass;
		Weight(vec2 _pos, float _mass) : pos(_pos), mass(_mass) {}
	};

	std::vector<Weight*> weights;
	float maxMass = 0.008f;
	float r0 = 0.005f;

public:
	SheetGeometry() { create(); }
	~SheetGeometry() {
		for (Weight* w : weights)
			delete w;
	}
	void addWeight(vec2 pos) {
		weights.push_back(new Weight(pos, maxMass));
		maxMass += 0.008f;
		create();
	}
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		X = U * 2.0f - 1.0f;
		Y = V * 2.0f - 1.0f;
		Z = Dnum2(0);
		for (Weight* w : weights) {
			Z = Z - Pow(Pow(Pow(X - w->pos.x, 2) + Pow(Y - w->pos.y, 2), 0.5f) + r0, -1) * w->mass;
		}
	}

	VertexData GenVertexDataXY(float x, float y) {
		return GenVertexData((x+1.0f)/2.0f, (y+1.0f)/2.0f);
	}

	bool checkWeightCollision(vec2 pos) {
		for (Weight* w : weights)
			if (dot(w->pos - pos, w->pos - pos) < 0.0001f)
				return true;
			return false;
	}
};



class Object {
public:
	PhongShader* shader;
	Material* material;
	Geometry* geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
public:
	Object(PhongShader* _shader, Material* _material, Geometry* _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		material = _material;
		geometry = _geometry;
	}

	virtual void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}

	void Draw(RenderState state) {
		mat4 M, Minv;
		SetModelingTransform(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		shader->Bind(state);
		geometry->Draw();
	}

	virtual void Animate(float tstart, float tend) {
	}
};

class Ball : public Object {
public:
	vec3 up;
	SheetGeometry *sheetGeometry;
	float energy;
	vec3 velocity;
	vec3 contactPoint;

	Ball(PhongShader* _shader, Material* _material, Geometry* _geometry, SheetGeometry* sheet) : Object( _shader, _material, _geometry) {
		this->sheetGeometry = sheet;
	}

	void calculateEnergy() {
		energy = 0.5f * dot(velocity, velocity);
	}

	virtual void Animate(float tstart, float tend) {
		VertexData vd = sheetGeometry->GenVertexDataXY(contactPoint.x, contactPoint.y);
		vec3 normal = normalize(vd.normal);
		vec3 a = (vec3(0, 0, -10) - dot(vec3(0, 0, -10), normal) * normal);
		velocity = velocity + a * (tend - tstart);
		contactPoint = contactPoint + (tend - tstart) * velocity;
		vd = sheetGeometry->GenVertexDataXY(contactPoint.x, contactPoint.y);
		velocity = velocity * energy / (0.5f * dot(velocity, velocity));

		contactPoint.z = vd.position.z;

		if (contactPoint.x > 1 - scale.x) contactPoint.x = -1 + scale.x;
		if (contactPoint.x < -1 + scale.x) contactPoint.x = 1 - scale.x;
		if (contactPoint.y > 1 - scale.y) contactPoint.y = -1 + scale.y;
		if (contactPoint.y < -1 + scale.y) contactPoint.y = 1 - scale.y;

		normal = normalize(vd.normal);
		translation = normal * scale.z + contactPoint;
		up = vd.normal;

	}

	vec3 getUP() {
		return up;
	}
};


class Sheet : public Object {
public:
	bool ballCameraOn = false;
private:
	SheetGeometry* sheetGeometry;
	std::vector<Ball*>* balls = new std::vector<Ball*>();
	PhongShader* phongShader = new PhongShader();
	Geometry* sphere = new Sphere();


	void removeBalls() {
		std::vector<Ball*>* temp = new std::vector<Ball*>();
		std::vector<Ball*> remove;
		for (Ball* ball : *balls){
			if (!sheetGeometry->checkWeightCollision(vec2(ball->contactPoint.x, ball->contactPoint.y)))
				temp->push_back(ball);
			else
				remove.push_back(ball);
		}
		for (Ball* ball : remove)
			delete ball;
		delete balls;
		balls = temp;
	}

	Material* createRandomMaterial() {
		Material* m = new Material();
		m->kd = vec3((float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX) / 2.0f;
		m->ks = vec3(4, 4, 4);
		m->ka = m->kd;
		m->shininess = 100;
		return m;
	}

	Ball* createBall(Material* m) {
		Ball* b = new Ball(phongShader, m, sphere, sheetGeometry);
		b->velocity = vec3(0, 0, 0);
		b->scale = vec3(0.05f, 0.05f, 0.05f);
		VertexData vd = GenVertexDataXY(-0.95f, -0.95f);
		b->contactPoint = vec3(-0.95, -0.95, vd.position.z);
		b->translation = b->contactPoint + normalize(vd.normal) * 0.05f;
		b->up = vd.normal;
		b->calculateEnergy();
		b->velocity = vec3(1, 1, 0);
		return b;
	}

public:
	Sheet(PhongShader* _shader, Material* _material) : Object(_shader, _material, new SheetGeometry()) {
		sheetGeometry = (SheetGeometry*)geometry;
		balls->push_back(createBall(createRandomMaterial()));
	}

	VertexData GenVertexDataXY(float x, float y) {
		return sheetGeometry->GenVertexDataXY(x, y);
	}

	void addWeight(vec2 pos) {
		sheetGeometry->addWeight(pos);
	}

	void startBall(vec3 velocity) {
		balls->back()->velocity = velocity;
		balls->back()->calculateEnergy();
		balls->push_back(createBall(createRandomMaterial()));

	}

	void Animate(float tstart, float tend) {
		for (int i = 0; i < balls->size()-1; i++)
			(*balls)[i]->Animate(tstart, tend);
		removeBalls();
	}

	void Draw(RenderState state) {
		Object::Draw(state);
		if (!ballCameraOn)
			(*balls)[0]->Draw(state);
		for (int i = 1; i < balls->size(); ++i)
			(*balls)[i]->Draw(state);
	}

	Ball* getFirstBall() {
		return balls->front();
	}
};
class Scene {
	std::vector<Object*> objects;
	Camera topViewCamera;
	Camera ballCamera;
	Light lights[2];
	
public:
	Sheet* sheet;
	bool normalCameraOn = true;
public:
	Scene() {}
	void Build() {
		SheetPhongShader* phongShader = new SheetPhongShader();

		Material* material0 = new Material;
		material0->kd = vec3(0.6f, 0.4f, 0.2f);
		material0->ks = vec3(2, 2, 2);
		material0->ka = vec3(0.1f, 0.1f, 0.1f);
		material0->shininess = 100;

		sheet = new Sheet(phongShader, material0);

		topViewCamera.wEye = vec3(0, 0, 1.3f);
		topViewCamera.wLookat = vec3(0, 0, 0);
		topViewCamera.wVup = vec3(0, 1, 0);
		ballCamera.fp = 0.01f;
		ballCamera.bp = 1;

		vec4 p1(0.8f, -0.5f, 2, 1), p2(-0.7f, 1.1f, 2, 1);
		lights[0] = Light(vec3(0.5f, 0.5f, 0.5f), vec3(1, 1, 1), p1, p2);
		lights[1] = Light(vec3(0.5f, 0.5f, 0.5f), vec3(1, 1, 1), p2, p1);
	}

	void Render() {
		RenderState state;
		if (normalCameraOn) {
			state.wEye = topViewCamera.wEye;
			state.V = topViewCamera.V();
			state.P = topViewCamera.P();
			sheet->ballCameraOn = false;
		}
		else {
			Ball* ball = sheet->getFirstBall();
			ballCamera.wEye = ball->translation;
			ballCamera.wLookat = ball->translation + ball->velocity*0.2f;
			ballCamera.wVup = normalize(ball->getUP());
			state.wEye = ballCamera.wEye;
			state.V = ballCamera.V();
			state.P = ballCamera.P();
			sheet->ballCameraOn = true;
		}
		state.lights = lights;
		sheet->Draw(state);
	}

	void Animate(float tstart, float tend) {
		sheet->Animate(tstart, tend);
		lights[0].Animate(tstart, tend);
		lights[1].Animate(tstart, tend);
	}
};

Scene scene;



void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
}


void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);						
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	scene.Render();
	glutSwapBuffers();									
}

void onKeyboard(unsigned char key, int pX, int pY) {
	scene.normalCameraOn = key == ' ' ? !scene.normalCameraOn : scene.normalCameraOn;
}


void onKeyboardUp(unsigned char key, int pX, int pY) { }


void onMouse(int button, int state, int pX, int pY) {
	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;
	if (state == GLUT_DOWN && button == GLUT_RIGHT_BUTTON) {

		scene.sheet->addWeight(vec2(cX, cY));
		return;
	}
	if (state == GLUT_DOWN && button == GLUT_LEFT_BUTTON) {
		scene.sheet->startBall(vec3((cX + 1.0f) / 2.0f, (cY + 1.0f) / 2.0f, 0));
		return;
	}
}

void onMouseMotion(int pX, int pY) {
	
}


void onIdle() {
	static float tend = 0;
	const float dt = 0.01f; 
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		scene.Animate(t, t + Dt);
	}
	glutPostRedisplay();
}
