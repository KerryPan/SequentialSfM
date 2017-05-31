#pragma once
#include "glut.h"

#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui_c.h>

#include <opencv2/core/core.hpp>



#include <cmath>

#include <algorithm>

#pragma comment(lib,"glut32.lib")

#define SIGN(x) ( (x)<0 ? -1:((x)>0?1:0 ) )
#define GL_PI 3.1415f
//---OpenGL ȫ�ֱ���

static GLfloat xRot = 0.0f;
static GLfloat yRot = 0.0f;

float xyzdata[480][640][3];
float texture[480][640][3];
int glWinWidth = 640, glWinHeight = 480;
int width = 640, height = 480;
double eyex, eyey, eyez, atx, aty, atz;  // eye* - �����λ�ã�at* - ע�ӵ�λ��

bool leftClickHold = false, rightClickHold = false;
int mx, my; 			// ��갴��ʱ�� OpenGL ���ڵ�����
int ry = 90, rx = 90;    // ��������ע�ӵ�Ĺ۲�Ƕ�
double mindepth, maxdepth;		// ������ݵļ�ֵ 
double radius = 6000.0;		// �������ע�ӵ�ľ���

void mouse(int button, int state, int x, int y);
void motion(int x, int y);

void SpecialKeys(int key, int x, int y);


void renderScene(void);
void reshape(int w, int h);
void load3dDataToGL(IplImage* xyz);
void loadTextureToGL(IplImage* img);

void run(IplImage* img1);