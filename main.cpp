#include "func.h"
//#include "newgl.h"

//sift特征点的向量是128维的，而cv::keyPoints只有6个维度
//keypoint只是保存了opencv的sift库检测到的特征点的一些基本信息，
//但sift所提取出来的特征向量其实不是在这个里面，特征向量通过SiftDescriptorExtractor 提取，
//结果放在一个Mat的数据结构中,即下面的descriptor。这个数据结构才真正保存了该特征点所对应的特征向量。

void main()
{
	vector<string> img_names;
	get_file_names("images", img_names);

	//本征矩阵,也就是相机的内参矩阵
	Mat K(Matx33d(
		2759.48, 0, 1520.69,
		0, 2764.16, 1006.81,
		0, 0, 1));

	vector<vector<KeyPoint>> key_points_for_all;
	vector<Mat> descriptor_for_all;
	vector<vector<Vec3b>> colors_for_all;
	vector<vector<DMatch>> matches_for_all;
	//提取所有图像的特征
	extract_features2(img_names, key_points_for_all, descriptor_for_all, colors_for_all);
	//对所有图像进行顺次的特征匹配
	match_features(key_points_for_all,descriptor_for_all, matches_for_all);

	vector<Point3f> structure;
	vector<vector<int>> correspond_struct_idx; //保存第i副图像中第j个特征点对应的structure中点的索引
	vector<Vec3b> colors;
	vector<Mat> rotations;
	vector<Mat> motions;

	//初始化结构（三维点云）
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


	//增量方式重建剩余的图像
	for (int i = 1; i < matches_for_all.size(); ++i)
	{
		vector<Point3f> object_points;
		vector<Point2f> image_points;
		Mat r, R, T;
		//Mat mask;

		//获取第i幅图像中匹配点对应的三维点，以及在第i+1幅图像中对应的像素点
		get_objpoints_and_imgpoints(
			matches_for_all[i], 
			correspond_struct_idx[i], 
			structure,
			key_points_for_all[i+1], 
			object_points,
			image_points
			);

		//求解变换矩阵
		solvePnPRansac(object_points, image_points, K, noArray(), r, T);
		//将旋转向量转换为旋转矩阵
		Rodrigues(r, R);
		//保存变换矩阵
		rotations.push_back(R);
		motions.push_back(T);

		vector<Point2f> p1, p2;
		vector<Vec3b> c1, c2;
		get_matched_points(key_points_for_all[i], key_points_for_all[i + 1], matches_for_all[i], p1, p2);
		get_matched_colors(colors_for_all[i], colors_for_all[i + 1], matches_for_all[i], c1, c2);

		//根据之前求得的R，T进行三维重建
		vector<Point3f> next_structure;
		reconstruct(K, rotations[i], motions[i], R, T, p1, p2, next_structure);

		//将新的重建结果与之前的融合
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

	//保存
	save_structure(".\\Viewer\\structure.yml", rotations, motions, structure, colors);

	save_text(structure);

	//glutInit(&argc, argv);

	cout << "nihao" << endl;
	cout << "第二次修改" << endl;
	waitKey();
}




