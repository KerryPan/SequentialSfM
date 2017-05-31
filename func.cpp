#include "func.h"

#include <fstream>
using namespace std;



void get_file_names(string dir_name, vector<string> & names)
{
	names.clear();
	tinydir_dir dir;
	tinydir_open(&dir, dir_name.c_str());

	while (dir.has_next)
	{
		tinydir_file file;
		tinydir_readfile(&dir, &file);
		if (!file.is_dir)
		{
			names.push_back(file.path);
		}
		tinydir_next(&dir);
	}
	tinydir_close(&dir);
}


void extract_features(
	vector<string>& image_names,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<Mat>& descriptor_for_all,
	vector<vector<Vec3b>>& colors_for_all
	)
{
	key_points_for_all.clear();
	descriptor_for_all.clear();
	Mat image;

	//��ȡͼ�񣬻�ȡͼ�������㣬������
	Ptr<Feature2D> sift = xfeatures2d::SIFT::create(0, 3, 0.04, 10);
	for (auto it = image_names.begin(); it != image_names.end(); ++it)
	{
		image = imread(*it);

		/************************************************************************/
		/*     resize      the picture                                                             */

		float stdWidth = 800, resizeScale = 1;//stdWidth can change
		if (image.cols > stdWidth * 1.2)
		{
			resizeScale = stdWidth / image.cols;
			Mat img;
			resize(image, img, Size(), resizeScale, resizeScale);
			
			image = img.clone();
			
		}

		imshow("l", image);
		/************************************************************************/
		if (image.empty()) continue;

		cout << "Extracing features: " << *it << endl;

		vector<KeyPoint> key_points;
		Mat descriptor;
		//ż�������ڴ����ʧ�ܵĴ���
		sift->detectAndCompute(image, noArray(), key_points, descriptor);

		//��������٣����ų���ͼ��
		if (key_points.size() <= 10) continue;

	//	imshow("out", descriptor);

		key_points_for_all.push_back(key_points);
		descriptor_for_all.push_back(descriptor);

		vector<Vec3b> colors(key_points.size());
		for (int i = 0; i < key_points.size(); ++i)
		{
			Point2f& p = key_points[i].pt;
			colors[i] = image.at<Vec3b>(p.y, p.x);
		}
		colors_for_all.push_back(colors);
	}
}


void extract_features2(
	vector<string>& image_names,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<Mat>& descriptor_for_all,
	vector<vector<Vec3b>>& colors_for_all
	)
{
	key_points_for_all.clear();
	descriptor_for_all.clear();
	Mat image;

	//��ȡͼ�񣬻�ȡͼ�������㣬������
	Ptr<Feature2D> surf = xfeatures2d::SURF::create(400, 4, 3, false, false);
	//��һ������hessianThreshold = 400 ��ͼ��Hessian�����б�ʽ����ֵ��ֵԽ��������������Խ�٣�Ҳ����ζ��������Խ�ȶ���

	for (auto it = image_names.begin(); it != image_names.end(); ++it)
	{
		image = imread(*it);
		if (image.empty()) continue;


	//	/************************************************************************/
	//	/*     resize      the picture                                                             */

	//	float stdWidth = 800, resizeScale = 1;//stdWidth can change
	//	if (image.cols > stdWidth * 1.2)
	//	{
	//		resizeScale = stdWidth / image.cols;
	//		Mat img;
	//		resize(image, img, Size(), resizeScale, resizeScale);

	//		image = img.clone();

	//	}

	///*	imshow("l", image);
	//	waitKey(0);*/
	//	/************************************************************************/

		cout << "Extracing features: " << *it << endl;

		vector<KeyPoint> key_points;
		Mat descriptor;
		//ż�������ڴ����ʧ�ܵĴ���
		surf->detectAndCompute(image, noArray(), key_points, descriptor);

		//��������٣����ų���ͼ��
		if (key_points.size() <= 10) continue;

		//	imshow("out", descriptor);

		key_points_for_all.push_back(key_points);
		descriptor_for_all.push_back(descriptor);

		vector<Vec3b> colors(key_points.size());
		for (int i = 0; i < key_points.size(); ++i)
		{
			Point2f& p = key_points[i].pt;
			colors[i] = image.at<Vec3b>(p.y, p.x);
		}
		colors_for_all.push_back(colors);
	}
}


int ratioTest(vector<vector<DMatch>>& matches) {

	int removed = 0;

	// for all matches
	for (vector<vector<DMatch>>::iterator matchIterator = matches.begin();
	matchIterator != matches.end(); ++matchIterator) {

		// if 2 NN has been identified
		if (matchIterator->size() > 1) {

			// check distance ratio  
			//ԭ��(*matchIterator)[0].distance ������ھ��룬(*matchIterator)[1].distance �Ǵν��ھ��룬����
			//(*matchIterator)[0].distance/(*matchIterator)[1].distance �϶�С��1
			//�������ںʹν��ڹ��ֽӽ�������ֵ�ӽ�1����ô�޷������Ǹ���������ƥ��㣬Ӧ���Ƴ��������
			//��������ratio����������ֵ�����������ʱ���Ƴ���ratio=0. 4������׼ȷ��Ҫ��ߵ�ƥ�䣻ratio = 0. 6������ƥ�����ĿҪ��Ƚ϶��ƥ�䣻 ratio = 0. 5��һ������¡�
			if ((*matchIterator)[0].distance / (*matchIterator)[1].distance > 0.6) {

				matchIterator->clear(); // remove match
				removed++;
			}

		}
		else { // does not have 2 neighbours

			matchIterator->clear(); // remove match
			removed++;
		}
	}

	return removed;
}

// Insert symmetrical matches in symMatches vector
void symmetryTest(const vector<vector<DMatch>>& matches1,
	const vector<vector<DMatch>>& matches2,
	vector<DMatch>& symMatches) {

	// for all matches image 1 -> image 2
	for (vector<vector<DMatch>>::const_iterator matchIterator1 = matches1.begin();
	matchIterator1 != matches1.end(); ++matchIterator1) {

		if (matchIterator1->size() < 2) // ignore deleted matches 
			continue;

		// for all matches image 2 -> image 1
		for (vector<vector<DMatch>>::const_iterator matchIterator2 = matches2.begin();
		matchIterator2 != matches2.end(); ++matchIterator2) {

			if (matchIterator2->size() < 2) // ignore deleted matches 
				continue;

			// Match symmetry test �Գ��Բ���
			if ((*matchIterator1)[0].queryIdx == (*matchIterator2)[0].trainIdx &&
				(*matchIterator2)[0].queryIdx == (*matchIterator1)[0].trainIdx) {

				// add symmetrical match  ��ӶԳƵ�ƥ��
				symMatches.push_back(DMatch((*matchIterator1)[0].queryIdx,
					(*matchIterator1)[0].trainIdx,
					(*matchIterator1)[0].distance));
				break; // next match in image 1 -> image 2
			}
		}
	}
}


void ransacTest(const vector<DMatch>& inputmatches,
	const vector<KeyPoint>& keypoints1,
	const vector<KeyPoint>& keypoints2,
	vector<DMatch>& outMatches) {

	// Convert keypoints into Point2f	
	vector<Point2f> points1, points2;
	double confidence = 0.99;
	double distance = 3.0;
	for (vector<DMatch>::const_iterator it = inputmatches.begin(); it != inputmatches.end(); ++it){

		// Get the position of left keypoints
		float x = keypoints1[it->queryIdx].pt.x;
		float y = keypoints1[it->queryIdx].pt.y;
		points1.push_back(Point2f(x, y));
		// Get the position of right keypoints
		x = keypoints2[it->trainIdx].pt.x;
		y = keypoints2[it->trainIdx].pt.y;
		points2.push_back(Point2f(x, y));
	}

	// Compute F matrix using RANSAC
	vector<uchar> inliers(points1.size(), 0);
	Mat fundemental = findFundamentalMat(
		Mat(points1), Mat(points2), // matching points
		inliers,      // match status (inlier or outlier)   ƥ��״̬
		CV_FM_RANSAC, // RANSAC method 
		distance,     // distance to epipolar line  ��ȷ�ĵ㵽���ߵ�������
		confidence);  // confidence probability ���Ÿ���

					  // extract the surviving (inliers) matches ��ȡͨ����ƥ��
	vector<uchar>::const_iterator itIn = inliers.begin();
	vector<DMatch>::const_iterator itM = inputmatches.begin();
	// for all matches
	for (; itIn != inliers.end(); ++itIn, ++itM) {

		if (*itIn) { // it is a valid match

			outMatches.push_back(*itM);
		}
	}	
}


void match_features(vector<vector<KeyPoint>>& key_points_for_all, vector<Mat>& descriptor_for_all, vector<vector<DMatch>>& matches_for_all)
{
	matches_for_all.clear();
	// n��ͼ������˳���� n-1 ��ƥ��
	// 1��2ƥ�䣬2��3ƥ�䣬3��4ƥ�䣬�Դ�����
	for (int i = 0; i < descriptor_for_all.size() - 1; ++i)
	{
		cout << "Matching images " << i << " - " << i + 1 << endl;
		vector<DMatch> matches;
		match_features(key_points_for_all[i], key_points_for_all[i + 1], descriptor_for_all[i], descriptor_for_all[i + 1], matches);
		matches_for_all.push_back(matches);
	}
}

//ƥ���������кܶ���ƥ�䣬Ϊ���ų���Щ��������ʹ����Ratio Test������
//��ʹ��KNN�㷨Ѱ�����������ƥ���2������������һ��������ƥ�������ڶ���������ƥ�����֮��С��ĳһ��ֵ���ͽ��ܸ�ƥ�䣬������Ϊ��ƥ�䡣
void match_features(vector<KeyPoint>& keypoints1,vector<KeyPoint>& keypoints2, Mat& descriptors1, Mat& descriptors2, vector<DMatch>& matches)
{
	vector<vector<DMatch>> knn_matches1;
	vector<vector<DMatch>> knn_matches2;
	BFMatcher matcher(NORM_L2);
	matcher.knnMatch(descriptors1, descriptors2, knn_matches1, 2); //ͼ1ÿ����������ͼ2�������������Ƶ�
	matcher.knnMatch(descriptors2, descriptors1, knn_matches2, 2); //ͼ1ÿ����������ͼ2�������������Ƶ�

	int removed = ratioTest(knn_matches1);
	removed = ratioTest(knn_matches2);
	
	//�Գ��Լ��
	symmetryTest(knn_matches1, knn_matches2, matches);

	
}



void get_matched_points(
	vector<KeyPoint>& p1,
	vector<KeyPoint>& p2,
	vector<DMatch> matches,
	vector<Point2f>& out_p1,
	vector<Point2f>& out_p2
	)
{
	out_p1.clear();
	out_p2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		out_p1.push_back(p1[matches[i].queryIdx].pt);
		out_p2.push_back(p2[matches[i].trainIdx].pt);
	}


}

void get_matched_colors(
	vector<Vec3b>& c1,
	vector<Vec3b>& c2,
	vector<DMatch> matches,
	vector<Vec3b>& out_c1,
	vector<Vec3b>& out_c2
	)
{
	out_c1.clear();
	out_c2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		out_c1.push_back(c1[matches[i].queryIdx]);
		out_c2.push_back(c2[matches[i].trainIdx]);
	}
}



//��Ҫ�����ڲξ���K������Ա任R��T
bool find_transform(Mat& K, vector<Point2f>& p1, vector<Point2f>& p2, Mat& R, Mat& T, Mat& mask)
{
	//�����ڲξ����ȡ����Ľ���͹������꣨�������꣩
	double focal_length = 0.5*(K.at<double>(0) + K.at<double>(4));
	Point2d principle_point(K.at<double>(2), K.at<double>(5));

	//����ƥ�����ȡ��������E��ʹ��RANSAC����һ���ų�ʧ���,mask���������1068�У�keypoints������1�У�E��3��3��
	Mat E = findEssentialMat(p1, p2, focal_length, principle_point, RANSAC, 0.999, 1.0, mask);
	if (E.empty()) return false;

	double feasible_count = countNonZero(mask);
	cout << (int)feasible_count << " -in- " << p1.size() << endl;
	//����RANSAC���ԣ�outlier��������50%ʱ������ǲ��ɿ���
	if (feasible_count <= 15 || (feasible_count / p1.size()) < 0.6)
		return false;

	//�ֽⱾ������E����ȡ��Ա任��R��T����3��3��
	//�õ������������ʹ����һ������recoverPose�Ա���������зֽ⣬�����������֮�����Ա任R��T��
	//ע�������T���ڵڶ������������ϵ�±�ʾ�ģ�Ҳ����˵���䷽��ӵڶ������ָ���һ�����������������ϵ���ڵ�������������ĳ��ȵ���1��
	int pass_count = recoverPose(E, p1, p2, R, T, focal_length, principle_point, mask);

	//ͬʱλ���������ǰ���ĵ������Ҫ�㹻��
	if (((double)pass_count) / feasible_count < 0.7)
		return false;

	return true;
}


void reconstruct(Mat& K, Mat& R1, Mat& T1, Mat& R2, Mat& T2, vector<Point2f>& p1, vector<Point2f>& p2, vector<Point3f>& structure)
{
	//���������ͶӰ����[R T]��triangulatePointsֻ֧��float��
	Mat proj1(3, 4, CV_32FC1); //3��4��
	Mat proj2(3, 4, CV_32FC1);  

	R1.convertTo(proj1(Range(0, 3), Range(0, 3)), CV_32FC1);
	T1.convertTo(proj1.col(3), CV_32FC1);

	R2.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1);
	T2.convertTo(proj2.col(3), CV_32FC1);
	
	Mat fK;
	K.convertTo(fK, CV_32FC1);
	proj1 = fK*proj1;
	proj2 = fK*proj2;

	//�����ؽ�
	Mat s; // 4xN ���������ϵ֮���ع�������
	triangulatePoints(proj1, proj2, p1, p2, s);

//	cout << "s.cols=" << endl << s.cols << endl;
//  s.cols����p�ĸ���1397

	structure.clear();
	structure.reserve(s.cols); //vectorͨ�� reserve() �������ض���С��ʱ�����ǰ�ָ���߽����������ڲ�������
	for (int i = 0; i < s.cols; ++i)
	{
		Mat_<float> col = s.col(i);  //colҲ�Ǹ�����
		col /= col(3);	//������꣬��Ҫ�������һ��Ԫ�ز�������������ֵ
		structure.push_back(Point3f(col(0), col(1), col(2)));
	}

	//structure ��Ԫ�صĸ�����p�ĸ������洢����point3f�����������
}

void maskout_points(vector<Point2f>& p1, Mat& mask)
{
	vector<Point2f> p1_copy = p1;
	p1.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
			p1.push_back(p1_copy[i]);
	}
}

void save_point3f(vector<Point3f>& point3d)
{


	ofstream outfile(".\\Viewer\\point3d.txt", ios::out);


	if (!outfile.is_open())
	{
		cout << " the file open fail" << endl;
		exit(1);
	}
	for (int i = 0; i<point3d.size(); i++)
	{

		outfile << point3d[i].x<<" ";

		outfile << point3d[i].y<<" ";

		outfile << point3d[i].z;

		outfile << "\n";
	}
	outfile.close();


}

void save_point2f(vector<Point2f>& point2d)
{


	ofstream outfile(".\\Viewer\\point2d.txt", ios::out);


	if (!outfile.is_open())
	{
		cout << " the file open fail" << endl;
		exit(1);
	}
	for (int i = 0; i<point2d.size(); i++)
	{

		outfile << point2d[i].x << " ";

		outfile << point2d[i].y;

		outfile << "\n";
	}
	outfile.close();


}

void maskout_colors(vector<Vec3b>& p1, Mat& mask)
{
	vector<Vec3b> p1_copy = p1;
	p1.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
			p1.push_back(p1_copy[i]);
	}
}


void init_structure(
	Mat K,
	vector<vector<KeyPoint>>& key_points_for_all,
	vector<vector<Vec3b>>& colors_for_all,
	vector<vector<DMatch>>& matches_for_all,
	vector<Point3f>& structure,
	vector<vector<int>>& correspond_struct_idx,
	vector<Vec3b>& colors,
	vector<Mat>& rotations,
	vector<Mat>& motions
	)
{
	//����ͷ����ͼ��֮��ı任����
	vector<Point2f> p1, p2;
	vector<Vec3b> c2;
	Mat R, T;	//��ת�����ƽ������
	Mat mask;	//mask�д�����ĵ����ƥ��㣬���������ʧ���




	get_matched_points(key_points_for_all[0], key_points_for_all[1], matches_for_all[0], p1, p2);
	get_matched_colors(colors_for_all[0], colors_for_all[1], matches_for_all[0], colors, c2);

	

	//cout << "1ƥ��������" << p1.size()<<endl;

	find_transform(K, p1, p2, R, T, mask);

	//cout << "p1 maskOutǰ�Ĵ�С��" << p1.size() << endl;
//	cout << "��matches.size():" << matches_for_all[0].size() << endl;
	
	//�Ѳ����ϵ�ƥ���ȥ��
	maskout_points(p1, mask);
	maskout_points(p2, mask);
	maskout_colors(colors, mask);

	

	//cout << "p1 maskOut��Ĵ�С��"<<p1.size()<<endl;


	Mat R0 = Mat::eye(3, 3, CV_64FC1);
	//  R0 =	[1, 0, 0
	//           0, 1, 0
	//           0, 0, 1]

	Mat T0 = Mat::zeros(3, 1, CV_64FC1);
//	T0 = [0,
//		  0,
//		  0]


//��ͷ����ͼ�������ά�ؽ�����������vector<Point3f>& structure
// ��һ���������ϵ��Ϊ��������ϵ���������ľ���ΪR0��T0
	reconstruct(K, R0, T0, R, T, p1, p2, structure);
	//����任����
	rotations = { R0, R };
	motions = { T0, T };

	//��correspond_struct_idx�Ĵ�С��ʼ��Ϊ��key_points_for_all��ȫһ��
	correspond_struct_idx.clear();   //vector<vector<int>>& correspond_struct_idx,
	correspond_struct_idx.resize(key_points_for_all.size());
	for (int i = 0; i < key_points_for_all.size(); ++i)
	{
		correspond_struct_idx[i].resize(key_points_for_all[i].size(), -1);
	}

	//��дͷ����ͼ��Ľṹ����correspond_struct_idx
	int idx = 0;
	vector<DMatch>& matches = matches_for_all[0];

	//cout << "matches.size():"<<matches.size()<<endl;

	for (int i = 0; i < matches.size(); ++i)
	{
		if (mask.at<uchar>(i) == 0)
			continue;

		correspond_struct_idx[0][matches[i].queryIdx] = idx;
		correspond_struct_idx[1][matches[i].trainIdx] = idx;
		++idx;
	}


	vector<Point3f> object_points;
	vector<Point2f> image_points;

	get_objpoints_and_imgpoints(
		matches_for_all[0],
		correspond_struct_idx[0],
		structure,
		key_points_for_all[0],
		object_points,
		image_points
	);

	save_point2f(image_points);
	save_point3f(object_points);

}


void get_objpoints_and_imgpoints(
	vector<DMatch>& matches,
	vector<int>& struct_indices,
	vector<Point3f>& structure,
	vector<KeyPoint>& key_points,
	vector<Point3f>& object_points,
	vector<Point2f>& image_points)
{
	object_points.clear();
	image_points.clear();

	
	for (int i = 0; i < matches.size(); ++i)
	{
		int query_idx = matches[i].queryIdx;
		int train_idx = matches[i].trainIdx;

		int struct_idx = struct_indices[query_idx];  //struct_indices �����ʵ���� correspond_struct_idx[0]
		if (struct_idx < 0) continue;

		object_points.push_back(structure[struct_idx]);
		//image_points.push_back(key_points[train_idx].pt);

		image_points.push_back(key_points[query_idx].pt);

		


	}

	//cout << "objectpoint;" << object_points.size() << endl;
	//cout << "image_points;" << image_points.size() << endl;
}

void fusion_structure(
	vector<DMatch>& matches,
	vector<int>& struct_indices,
	vector<int>& next_struct_indices,
	vector<Point3f>& structure,
	vector<Point3f>& next_structure,
	vector<Vec3b>& colors,
	vector<Vec3b>& next_colors
	)
{
	for (int i = 0; i < matches.size(); ++i)
	{
		int query_idx = matches[i].queryIdx;
		int train_idx = matches[i].trainIdx;

		int struct_idx = struct_indices[query_idx];
		if (struct_idx >= 0) //���õ��ڿռ����Ѿ����ڣ������ƥ����Ӧ�Ŀռ��Ӧ����ͬһ��������Ҫ��ͬ
		{
			next_struct_indices[train_idx] = struct_idx;
			continue;
		}

		//���õ��ڿռ����Ѿ����ڣ����õ���뵽�ṹ�У������ƥ���Ŀռ��������Ϊ�¼���ĵ������
		structure.push_back(next_structure[i]);


		colors.push_back(next_colors[i]);
		struct_indices[query_idx] = next_struct_indices[train_idx] = structure.size() - 1;
	
	}

	//cout << structure.size();
}


void save_structure(string file_name, vector<Mat>& rotations, vector<Mat>& motions, vector<Point3f>& structure, vector<Vec3b>& colors)
{
	int n = (int)rotations.size();

	FileStorage fs(file_name, FileStorage::WRITE);
	fs << "Camera Count" << n;
	fs << "Point Count" << (int)structure.size();

	fs << "Rotations" << "[";
	for (size_t i = 0; i < n; ++i)
	{
		fs << rotations[i];
	}
	fs << "]";

	fs << "Motions" << "[";
	for (size_t i = 0; i < n; ++i)
	{
		fs << motions[i];
	}
	fs << "]";

	fs << "Points" << "[";
	for (size_t i = 0; i < structure.size(); ++i)
	{
		fs << structure[i];
	}
	fs << "]";

	fs << "Colors" << "[";
	for (size_t i = 0; i < colors.size(); ++i)
	{
		fs << colors[i];
	}
	fs << "]";

	fs.release();
}


void save_text(vector<Point3f>& structure)
{
	

	ofstream outfile(".\\Viewer\\newpoints.txt",ios::out);
	

	if (!outfile.is_open())
	{
		cout << " the file open fail" << endl;
		exit(1);
	}
	for (int i = 0; i<structure.size(); i++)
	{
		
		outfile << structure[i];
		
		outfile << "\n";
	}
	outfile.close();

	
}

