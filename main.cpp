#include "func.h"
//#include "newgl.h"

//sift�������������128ά�ģ���cv::keyPointsֻ��6��ά��
//keypointֻ�Ǳ�����opencv��sift���⵽���������һЩ������Ϣ��
//��sift����ȡ����������������ʵ������������棬��������ͨ��SiftDescriptorExtractor ��ȡ��
//�������һ��Mat�����ݽṹ��,�������descriptor��������ݽṹ�����������˸�����������Ӧ������������

void main()
{
	vector<string> img_names;
	get_file_names("images", img_names);

	//��������,Ҳ����������ڲξ���
	Mat K(Matx33d(
		2759.48, 0, 1520.69,
		0, 2764.16, 1006.81,
		0, 0, 1));

	vector<vector<KeyPoint>> key_points_for_all;
	vector<Mat> descriptor_for_all;
	vector<vector<Vec3b>> colors_for_all;
	vector<vector<DMatch>> matches_for_all;
	//��ȡ����ͼ�������
	extract_features2(img_names, key_points_for_all, descriptor_for_all, colors_for_all);
	//������ͼ�����˳�ε�����ƥ��
	match_features(key_points_for_all,descriptor_for_all, matches_for_all);

	vector<Point3f> structure;
	vector<vector<int>> correspond_struct_idx; //�����i��ͼ���е�j���������Ӧ��structure�е������
	vector<Vec3b> colors;
	vector<Mat> rotations;
	vector<Mat> motions;

	//��ʼ���ṹ����ά���ƣ�
	init_structure(
		K,
		key_points_for_all,
		colors_for_all,
		matches_for_all,
		structure,
		correspond_struct_idx,
		colors,
		rotations,
		motions
	);


	//������ʽ�ؽ�ʣ���ͼ��
	for (int i = 1; i < matches_for_all.size(); ++i)
	{
		vector<Point3f> object_points;
		vector<Point2f> image_points;
		Mat r, R, T;
		//Mat mask;

		//��ȡ��i��ͼ����ƥ����Ӧ����ά�㣬�Լ��ڵ�i+1��ͼ���ж�Ӧ�����ص�
		get_objpoints_and_imgpoints(
			matches_for_all[i], 
			correspond_struct_idx[i], 
			structure,
			key_points_for_all[i+1], 
			object_points,
			image_points
			);

		//���任����
		solvePnPRansac(object_points, image_points, K, noArray(), r, T);
		//����ת����ת��Ϊ��ת����
		Rodrigues(r, R);
		//����任����
		rotations.push_back(R);
		motions.push_back(T);

		vector<Point2f> p1, p2;
		vector<Vec3b> c1, c2;
		get_matched_points(key_points_for_all[i], key_points_for_all[i + 1], matches_for_all[i], p1, p2);
		get_matched_colors(colors_for_all[i], colors_for_all[i + 1], matches_for_all[i], c1, c2);

		//����֮ǰ��õ�R��T������ά�ؽ�
		vector<Point3f> next_structure;
		reconstruct(K, rotations[i], motions[i], R, T, p1, p2, next_structure);

		//���µ��ؽ������֮ǰ���ں�
		fusion_structure(
			matches_for_all[i], 
			correspond_struct_idx[i], 
			correspond_struct_idx[i + 1],
			structure, 
			next_structure,
			colors,
			c1
			);
	}

	//����
	save_structure(".\\Viewer\\structure.yml", rotations, motions, structure, colors);

	save_text(structure);

	//glutInit(&argc, argv);

	cout << "nihao" << endl;
	cout << "�ڶ����޸�" << endl;
	waitKey();
}




