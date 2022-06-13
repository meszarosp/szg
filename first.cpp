//=============================================================================================
// Hazi feladat: Graf rajzolo fokusszal. Ervenyes 2019. osztol.
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
// Nev    : Meszaros Péter
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

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	layout(location = 0) in vec3 vp;	// Varying input: vp = vertex position is expected in attrib array 0
	layout(location = 1) in vec2 vertexUV;

	out vec2 texCoord;

	void main() {
		texCoord = vertexUV;
		gl_Position = vec4(vp.x/vp.z, vp.y/vp.z, 0, 1);		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;
	uniform bool edge;
	uniform sampler2D textureUnit;

	in vec2 texCoord;
	out vec4 outColor;		// computed color of the current pixel


	void main() {
		outColor = edge ? vec4(color, 1.0) : texture(textureUnit, texCoord);
	}
)";

float Lorentz(vec3 p, vec3 q) {
	return p.x * q.x + p.y * q.y - p.z * q.z;
}

float distance(vec3 p, vec3 q) {
	return acoshf(-(p.x * q.x + p.y * q.y - p.z * q.z));
}

vec3 normalizeHyperbolic(vec3 v) {
	float L = Lorentz(v, v);
	if (L <= 0.0f)
		return v;

	return v / sqrtf(L);
}

vec3 mirror(vec3 p, vec3 m) {
	float d = distance(p, m);
	if (d <= 0.0f)
		return p;
	vec3 v = (m - p * coshf(d)) / sinhf(d);
	return p * coshf(2 * d) + v * sinhf(2 * d);
}

vec3 translateLike(vec3 p, vec3 start, vec3 end) {
	float d = distance(start, end);
	if (d <= 0.0f)
		return p;
	vec3 v = (end - start * coshf(d)) / sinhf(d);
	vec3 m1 = start * coshf(d / 4.0f) + v * sinhf(d / 4.0f);
	vec3 m2 = start * coshf(3.0f * d / 4.0f) + v * sinhf(3.0f * d / 4.0f);
	vec3 q = mirror(mirror(p, m1), m2);
	q.z = sqrtf(q.x * q.x + q.y * q.y + 1);
	return q;
}

vec3 getTangent(vec3 p, vec3 q) {
	float d = distance(p, q);
	return  (q - p * coshf(d)) / sinhf(d);
}

int nCirclePoints = 50;
vec3 buttonDownPos( 0.0f, 0.0f, 1.0f);
vec3 currentPos(0.0f, 0.0f, 1.0f);
bool buttonDown = false;
GPUProgram gpuProgram;
int nVertices = 50;

class Node {
	vec3 point;
	vec3 nextPos;
	std::vector<Node*> neighbours;
	std::vector<Node*> notNeighbours;

	std::vector<vec3> unitCircle;
	std::vector<vec2> unitCircleUV;
	Texture* texture;

	vec3 F = vec3(0.0f, 0.0f, 0.0f);
	vec3 velocity = vec3(0.0f, 0.0f, 0.0f);
	vec3 q = vec3(0.0f, 0.0f, 1.0f);
	float velocityLength = 0.0f;
private:
	void generateTexture() {
		vec4 color1 = vec4(float(rand()) / RAND_MAX, float(rand()) / RAND_MAX, float(rand()) / RAND_MAX, 1.0f);
		vec4 color2 = vec4(float(rand()) / RAND_MAX, float(rand()) / RAND_MAX, float(rand()) / RAND_MAX, 1.0f);
		std::vector<vec4> image;
		for (int y = 1; y <= 32; y++) {
			for (int x = 1; x <= 32; x++) {
				if (x <= 8 || y <= 8 || x >= 24 || y >= 24)
					image.push_back(vec4(0.65f, 0.65f, 0.65f, 1.0f));
				else if (9 <= y && y <= 23 && 9 <= x && x <= 15)
					image.push_back(color1);
				else if (9 <= y && y <= 23 && 16 <= x && x <= 23)
					image.push_back(color2);
			}
		}

		texture = new Texture(32, 32, image);
	}
public:
	Node(vec3 _point) :point(_point), nextPos(_point) {
		generateTexture();
		float radius = 0.06;
		for (int i = 0; i < nCirclePoints; i++) {
			float t = 2 * M_PI * i / nCirclePoints;
			vec2 uv = vec2(cosf(t), sinf(t));
			unitCircleUV.push_back((uv + vec2(1, 1)) * 0.5f);
			float w = sqrtf(uv.x * uv.x + uv.y * uv.y + 1);
			vec3 p(uv.x, uv.y, w);
			vec3 origo(0.0f, 0.0f, 1.0f);
			vec3 v = getTangent(origo, p);
			vec3 q = origo * coshf(radius) + v * sinhf(radius);
			unitCircle.push_back(q);
		}
	}

	vec3 getPoint() {
		return point;
	}
	void addNeighbour(Node* node) {
		neighbours.push_back(node);
		for (auto iter = notNeighbours.begin(); iter != notNeighbours.end(); ++iter) {
			if (node == *iter) {
				notNeighbours.erase(iter);
				break;
			}
		}
	}
	void addNotNeighbour(Node* node) {
		notNeighbours.push_back(node);
	}
	void draw(unsigned int vao, unsigned int vbo[]) {
		glBindVertexArray(vao);


		std::vector<vec3> circleData;
		for (vec3 p : unitCircle) {
			vec3 point2 = point;
			if (buttonDown)
				point2 = translateLike(point, buttonDownPos, currentPos);
			p = translateLike(p, vec3(0.0f, 0.0f, 1.0f), point2);
			circleData.push_back(p);
		}
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		glBufferData(GL_ARRAY_BUFFER, circleData.size() * 3 * sizeof(float), &circleData[0], GL_DYNAMIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
		glBufferData(GL_ARRAY_BUFFER, unitCircleUV.size() * 2 * sizeof(float), &unitCircleUV[0], GL_STATIC_DRAW);

		gpuProgram.setUniform(*texture, "textureUnit");

		glDrawArrays(GL_TRIANGLE_FAN, 0, nCirclePoints);
	}

	void finalizeTranslation() {
		point = translateLike(point, buttonDownPos, currentPos);
	}

	void step() {
		point = nextPos;
	}

	void heuristic() {
		nextPos = vec3(0.0f, 0.0f, 0.0f);
		for (Node* node : neighbours) {
			//Sulyozassal
			nextPos = nextPos + 10.0f*node->getPoint();
		}
		for (Node* node : notNeighbours) {
			nextPos = nextPos - 0.5f*node->getPoint();
		}
		nextPos = nextPos/(neighbours.size() + notNeighbours.size());
		nextPos.z = sqrtf(nextPos.x * nextPos.x + nextPos.y * nextPos.y + 1);
	}

	void simulate(float dt) {
		float prefDist = 0.5f;
		float d = distance(point, vec3(0.0f, 0.0f, 1.0f));
		velocity = getTangent(point, q) * velocityLength;
		F = getTangent(point, vec3(0.0f, 0.0f, 1.0f)) * d * 3.0f -velocity * 5.0f;
		for (Node* node : neighbours) {
			d = distance(point, node->getPoint());
			vec3 tangent = getTangent(point, node->getPoint());
			if (d - prefDist > 0.0f)
				tangent = tangent * (d - prefDist);
			else
				tangent = tangent * (d - prefDist) * 10.0f;
			F = F + tangent;
		}
		for (Node* node : notNeighbours) {
			d = distance(point, node->getPoint());
			F = F - getTangent(point, node->getPoint()) * 0.12f/(d);
		}

		if (Lorentz(F,F) < 0.0001f) {
			nextPos = point;
			return;
		}
		velocity = velocity + F*dt;
		velocityLength = Lorentz(velocity, velocity);
		velocity = normalizeHyperbolic(velocity);

		q = point * coshf(10.0f) + velocity * sinhf(10.0f); // egy tavoli pont az irany megorzesehez
		nextPos = point * coshf(dt) + velocity * sinhf(dt);
		nextPos.z = sqrtf(nextPos.x * nextPos.x + nextPos.y * nextPos.y + 1); //korrekcio a pontatlansagok miatt
	}
};

int side =	2.5f;
float density = 0.05f;

class Graph {
	unsigned int vao;
	unsigned int vbo[2];
	std::vector<Node*> nodes;
	std::vector<std::pair<Node*, Node*> > edges;
private:
	void generate() {
		for (int i = 0; i < nVertices; i++) {
			float x = ((float(rand())) / RAND_MAX * 2.0f - 1.0f) * side;
			float y = ((float(rand())) / RAND_MAX * 2.0f - 1.0f) * side;

			float w = sqrtf(x * x + y * y + 1);
			nodes.push_back(new Node(vec3(x, y, w)));
		}

		for (int i = 0; i < nVertices; i++) {
			for (int j = 0; j < nVertices; j++) {
				if (i != j)
					nodes[i]->addNotNeighbour(nodes[j]);
			}
		}

		int maxEdges = (nVertices * (nVertices - 1) / 2) * density;
		while (edges.size() < maxEdges) {
			std::pair<Node*, Node*> edge = { nodes[rand() % nVertices], nodes[rand() % nVertices] };
			if (edge.first == edge.second) continue;
			bool found = false;
			for (int i = 0; i < edges.size() && !found; i++) {
				if (edges[i].first == edge.first && edges[i].second == edge.second ||
					edges[i].first == edge.second && edges[i].second == edge.first)
					found = true;
			}
			if (!found) {
				edges.push_back(edge);
				edge.first->addNeighbour(edge.second);
				edge.second->addNeighbour(edge.first);
			}
		}
	}

	void drawEdges() {
		std::vector<vec3> edgeData;
		if (buttonDown) {
			for (auto edge : edges) {
				edgeData.push_back(translateLike(edge.first->getPoint(), buttonDownPos, currentPos));
				edgeData.push_back(translateLike(edge.second->getPoint(), buttonDownPos, currentPos));
			}
		}
		else {
			for (auto edge : edges) {
				edgeData.push_back(edge.first->getPoint());
				edgeData.push_back(edge.second->getPoint());
			}
		}

		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, 1.0f, 1.0f, 0.0f);
		location = glGetUniformLocation(gpuProgram.getId(), "edge");
		glUniform1i(location, 1);

		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		glBufferData(GL_ARRAY_BUFFER, edgeData.size() * 3 * sizeof(float), &edgeData[0], GL_DYNAMIC_DRAW);

		glDrawArrays(GL_LINES, 0, edgeData.size());
	}
public:
	void create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(2, vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);
		generate();
	}
	void draw() {
		drawEdges();
		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, 0.0f, 1.0f, 0.0f);
		location = glGetUniformLocation(gpuProgram.getId(), "edge");
		glUniform1i(location, 0);
		for (Node* node : nodes) {
			node->draw(vao, vbo);
		}
	}
	void finalizeTranslation() {
		for (Node* node : nodes)
			node->finalizeTranslation();
	}

	void simulate(float dt) {
		for (Node* node : nodes)
			node->simulate(dt);
		for (Node* node : nodes)
			node->step();
	}

	void heuristic() {
		for (Node* node : nodes)
			node->heuristic();
		for (Node* node : nodes)
			node->step();
	}
};

Graph graph;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	srand(0);
	graph.create();
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
	//glutPostRedisplay();
}

void onDisplay() {
	glClearColor(0, 0, 0, 0);     
	glClear(GL_COLOR_BUFFER_BIT);
	graph.draw();
	glutSwapBuffers();
}

bool animate = false;
long startingTime = 0;

void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == ' ') {
		animate = true;
		startingTime = glutGet(GLUT_ELAPSED_TIME);
		graph.heuristic();
		glutPostRedisplay();
	}
}

void onKeyboardUp(unsigned char key, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;
	if (buttonDown) {
		float w = sqrtf(1 - cX * cX - cY * cY);
		currentPos = vec3(cX / w, cY / w, 1 / w);
		glutPostRedisplay();
	}
}

void onMouse(int button, int state, int pX, int pY) {
	float cX = 2.0f * pX / windowWidth - 1;
	float cY = 1.0f - 2.0f * pY / windowHeight;

	if (button == GLUT_RIGHT_BUTTON && state == GLUT_UP) {
		buttonDown = false;
		graph.finalizeTranslation();
		glutPostRedisplay();
	}

	if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
		buttonDown = true;
		float w = sqrtf(1 - cX * cX - cY * cY);
		buttonDownPos = vec3(cX / w, cY / w, 1 / w);
	}
}

long lastTime = 0;


void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME);
	if (time - startingTime > 500)
		animate = false;
	if (time - lastTime > 16 && animate) {
		lastTime = time;
		while (time - lastTime < 16) {
			graph.simulate(0.01f);
			time = glutGet(GLUT_ELAPSED_TIME);
		}
		glutPostRedisplay();
	}
	
}
