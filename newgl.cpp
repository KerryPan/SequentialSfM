// 
#include "newgl.h"

/************************************************************************/
/*                                           OpenGL��Ӧ����                                                 */
/************************************************************************/

//////////////////////////////////////////////////////////////////////////
// ��갴����Ӧ����
void mouse(int button, int state, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON)
	{
		if (state == GLUT_DOWN)
		{
			leftClickHold = true;
		}
		else
		{
			leftClickHold = false;
		}
	}

	if (button == GLUT_RIGHT_BUTTON)
	{
		if (state == GLUT_DOWN)
		{
			rightClickHold = true;
		}
		else
		{
			rightClickHold = false;
		}
	}
}

//////////////////////////////////////////////////////////////////////////
// ����˶���Ӧ����
void motion(int x, int y)
{
	int rstep = 5;
	if (leftClickHold == true)
	{
		if (abs(x - mx) > abs(y - my))
		{
			rx += SIGN(x - mx)*rstep;
		}
		else
		{
			ry -= SIGN(y - my)*rstep;
		}

		mx = x;
		my = y;
		glutPostRedisplay();
	}

	if (rightClickHold == true)
	{
		radius += SIGN(y - my)*100.0;
		radius = std::max(radius, 100.0);
		mx = x;
		my = y;
		glutPostRedisplay();
	}
}

//////////////////////////////////////////////////////////////////////////
// ��άͼ����ʾ��Ӧ����
void renderScene(void)
{
	// clear screen and depth buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	// Reset the coordinate system before modifying 
	glLoadIdentity();
	// set the camera position
	atx = 0.0f;
	aty = 0.0f;
	atz = (mindepth - maxdepth) / 2.0f;
	eyex = atx + radius * sin(GL_PI * ry / 180.0f) * cos(GL_PI * rx / 180.0f);
	eyey = aty + radius * cos(GL_PI * ry / 180.0f);
	eyez = atz + radius * sin(GL_PI * ry / 180.0f) * sin(GL_PI * rx / 180.0f);
	gluLookAt(eyex, eyey, eyez, atx, aty, atz, 0.0, 1.0, 0.0);
	glRotatef(0, 0, 1, 0);
	glRotatef(-180, 1, 0, 0);

// �Ե������ݽ������ǻ�
// �ο��ԣ�http://www.codeproject.com/KB/openGL/OPENGLTG.aspx
// we are going to loop through all of our terrain's data points,
// but we only want to draw one triangle strip for each set along the x-axis.
	for (int i = 0; i < height; i++)
	{
		glBegin(GL_TRIANGLE_STRIP);
		for (int j = 0; j < width; j++)
		{
			// for each vertex, we calculate the vertex color, 
			// we set the texture coordinate, and we draw the vertex.
			/*
			the vertexes are drawn in this order:

			0  ---> 1
			/
			/
			/
			2  ---> 3
			*/

			// draw vertex 0
			glTexCoord2f(0.0f, 0.0f);
			glColor3f(texture[i][j][0] / 255.0f, texture[i][j][1] / 255.0f, texture[i][j][2] / 255.0f);
			glVertex3f(xyzdata[i][j][0], xyzdata[i][j][1], xyzdata[i][j][2]);

			// draw vertex 1
			glTexCoord2f(1.0f, 0.0f);
			glColor3f(texture[i + 1][j][0] / 255.0f, texture[i + 1][j][1] / 255.0f, texture[i + 1][j][2] / 255.0f);
			glVertex3f(xyzdata[i + 1][j][0], xyzdata[i + 1][j][1], xyzdata[i + 1][j][2]);

			// draw vertex 2
			glTexCoord2f(0.0f, 1.0f);
			glColor3f(texture[i][j + 1][0] / 255.0f, texture[i][j + 1][1] / 255.0f, texture[i][j + 1][2] / 255.0f);
			glVertex3f(xyzdata[i][j + 1][0], xyzdata[i][j + 1][1], xyzdata[i][j + 1][2]);

			// draw vertex 3
			glTexCoord2f(1.0f, 1.0f);
			glColor3f(texture[i + 1][j + 1][0] / 255.0f, texture[i + 1][j + 1][1] / 255.0f, texture[i + 1][j + 1][2] / 255.0f);
			glVertex3f(xyzdata[i + 1][j + 1][0], xyzdata[i + 1][j + 1][1], xyzdata[i + 1][j + 1][2]);
		}
		glEnd();
	}
	// enable blending
	glEnable(GL_BLEND);

	// enable read-only depth buffer
	glDepthMask(GL_FALSE);

	// set the blend function to what we use for transparency
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);

	// set back to normal depth buffer mode (writable)
	glDepthMask(GL_TRUE);

	// disable blending
	glDisable(GL_BLEND);

	/* 	float x,y,z;
	// ����ͼ�����
	glPointSize(1.0);
	glBegin(GL_POINTS);
	for (int i=0;i<height;i++){
	for (int j=0;j<width;j++){
	// color interpolation
	glColor3f(texture[i][j][0]/255, texture[i][j][1]/255, texture[i][j][2]/255);
	x= xyzdata[i][j][0];
	y= xyzdata[i][j][1];
	z= xyzdata[i][j][2];
	glVertex3f(x,y,z);
	}
	}
	glEnd(); */

	glFlush();
	glutSwapBuffers();
}

//////////////////////////////////////////////////////////////////////////
// ���ڱ仯ͼ���ع���Ӧ����
void reshape(int w, int h)
{
	glWinWidth = w;
	glWinHeight = h;
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45, (GLfloat)w / (GLfloat)h, 1.0, 15000.0);
	glMatrixMode(GL_MODELVIEW);
}

//////////////////////////////////////////////////////////////////////////
// ������ά��������
void load3dDataToGL(IplImage* xyz)
{
	CvScalar s;
	//accessing the image pixels
	for (int i = 0; i<height; i++)
	{
		for (int j = 0; j<width; j++)
		{
			s = cvGet2D(xyz, i, j);			// s.val[0] = x, s.val[1] = y, s.val[2] = z
			xyzdata[i][j][0] = s.val[0];
			xyzdata[i][j][1] = s.val[1];
			xyzdata[i][j][2] = s.val[2];
		}
	}
}

//////////////////////////////////////////////////////////////////////////
// ����ͼ����������
void loadTextureToGL(IplImage* img)
{
	CvScalar ss;
	//accessing the image pixels
	for (int i = 0; i<height; i++)
	{
		for (int j = 0; j<width; j++)
		{
			ss = cvGet2D(img, i, j);			// ss.val[0] = red, ss.val[1] = green, ss.val[2] = blue
			texture[i][j][2] = ss.val[0];
			texture[i][j][1] = ss.val[1];
			texture[i][j][0] = ss.val[2];
		}
	}
}

//OpenGL---���ܼ������������Ӧ����
void SpecialKeys(int key, int x, int y)
{
	if (key == GLUT_KEY_UP)
		xRot -= 5.0f;

	if (key == GLUT_KEY_DOWN)
		xRot += 5.0f;

	if (key == GLUT_KEY_LEFT)
		yRot -= 5.0f;

	if (key == GLUT_KEY_RIGHT)
		yRot += 5.0f;

	if (key > 356.0f)
		xRot = 0.0f;

	if (key < -1.0f)
		xRot = 355.0f;

	if (key > 356.0f)
		yRot = 0.0f;

	if (key < -1.0f)
		yRot = 355.0f;

	// Refresh the Window
	glutPostRedisplay();
}

void run(IplImage* img1)
{
	CvSize imageSize = { 0,0 };
	imageSize = cvGetSize(img1);
	IplImage* Image2 = cvCreateImageHeader(imageSize, IPL_DEPTH_32F, 3);

	CvMat* result3DImage = cvCreateMat(imageSize.height,
		imageSize.width, CV_32FC3);

	cvGetImage(result3DImage, Image2);
	//glutInit(int argc, char* argv[]);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_SINGLE | GLUT_RGBA);
	glutInitWindowPosition(10, 390);
	glutInitWindowSize(450, 390);
	glutCreateWindow("3D disparity image");


	//////////////////////////////////////////////////////////////////////////
	// OpenGL��ʾ
	//CvMat* img3dIpl=cvCreateMat(imageSize.height,imageSize.width,CV_32FC3);
	//cvConvert(result3DImage,img3dIpl);
	//img3dIpl = result3DImage;
	//CvMat* img1roi=cvCreateMat(imageSize.height,imageSize.width,CV_32FC3);
	//cvConvert(img1,img1roi);
	height = imageSize.height;
	width = imageSize.width;
	load3dDataToGL(Image2);            // ������ά��������
	loadTextureToGL(img1);               // ������������
	glutReshapeFunc(reshape);            // ���ڱ仯ʱ�ع�ͼ��
	glutDisplayFunc(renderScene);         // ��ʾ��άͼ��
	glutSpecialFunc(SpecialKeys);             // ��Ӧ�����������Ϣ
	glutPostRedisplay();                  // ˢ�»��棨���ô�������ܶ�̬����ͼ��
										  //glutMainLoopEvent();
	glutMainLoop();
	cvReleaseImage(&Image2);
	
}