//
// Created by anreydron on 20.04.2022.
//
#include "libprj.h"

#include <cmath>

Mat ColorCorrector::Dispersion(const Mat& source, const Size& size)
{
	cv::Mat copy(source.size(),CV_32FC3);
	cv::Mat buffer(source.size(), CV_32FC3);
	cv::Mat rgb[3];
	cv::Mat rgb_[3];
	source.convertTo(copy, CV_32FC3, 1.0/255.0, 0.0);
	source.convertTo(buffer, CV_32FC3, 1.0/255.0, 0.0);
	cv::split(buffer, rgb_);
	cv::split(copy, rgb);
	cv::Mat answer(source.size(),CV_32FC1, cv::Scalar(1.0));
	for(int i = 0; i < 3; ++i)
	{
		cv::multiply(rgb[i], rgb[i], rgb[i]);
		cv::GaussianBlur(rgb[i], rgb[i], size, 0, 0);
		cv::GaussianBlur(rgb_[i], rgb_[i], size, 0,0);
		cv::multiply(rgb_[i], rgb_[i], rgb_[i]);
		rgb[i] = rgb[i] - rgb_[i];
		cv::multiply(answer,rgb[i],answer);
	}
	cv::pow(answer, 1.0/3.0,answer);
	return answer;
}
ColorCorrector::ConvexHullInfo ColorCorrector::CHull(const std::vector<Vector3<float>>& points)
{
	QuickHull<float> qh;
	auto mesh = qh.getConvexHull(points, true, true);
	auto buffer = mesh.getIndexBuffer();
	return ColorCorrector::ConvexHullInfo{points, mesh, buffer};
}
float ColorCorrector::Det2x2(const float& l1l, const float& l1r, const float& l2l, const float& l2r)
{
	return l1l*l2r - l1r*l2l;
}
float ColorCorrector::R(std::pair<int,int>& lf , std::pair<int,int>& rh)
{
	return float((sqrt(pow(lf.first - rh.first, 2) + pow(lf.second - rh.second,2))));
}
float ColorCorrector::Det3x3(const Vector3<float>& l1, const Vector3<float>& l2, const Vector3<float>& l3)
{
	float volume = (l1.x * Det2x2(l2.y, l2.z, l3.y, l3.z)  -
		l1.y * Det2x2(l2.x,l2.z, l3.x, l3.z) +
		l1.z * Det2x2(l2.x, l2.y, l3.x, l3.y));
	return volume;
}
float ColorCorrector::TetraVolume(const Vector3<float>& r1,const Vector3<float>& r2,const Vector3<float>& r3,const Vector3<float>& r4)
{
	Vector3<float> l1 = r2 - r1;
	Vector3<float> l2 = r3 - r1;
	Vector3<float> l3 = r4 - r1;
	float volume = float(1.0/6.0) * Det3x3(l1,l2,l3);
	return volume > 0 ? volume : -volume;
}
float ColorCorrector::Volume(const std::vector<unsigned long>& indices, const std::vector<Vector3<float>>& pointCloud)
{
	float volume = 0.0;
	Vector3<float> cm{0,0,0};
	for(auto& index : indices)
		cm += pointCloud[index];
	cm /= float(indices.size());
	for(int i = 0; i < indices.size(); i+=3)
		volume += TetraVolume(cm,pointCloud[indices[i]], pointCloud[indices[i+1]], pointCloud[indices[i+2]]);
	return volume;
}
float ColorCorrector::ScalarProduct(const Vector3<float>& lh, const Vector3<float>& rh)
{
	return lh.x*rh.x + lh.y*rh.y + lh.z*rh.z;
}
Vector3<float> ColorCorrector::VectorProduct(const Vector3<float>& lh, const Vector3<float>& rh)
{
	return Vector3<float> { Det2x2(lh.y,lh.z,rh.y,rh.z), -Det2x2(lh.x,lh.z, rh.x,rh.z), Det2x2(lh.x,lh.y,rh.x,rh.y)};
}
void ColorCorrector::DropDotsToFile(const std::string& filename, const Mat& src)
{
	std::fstream file;
	file.open(filename.c_str(),  std::ios::out);
	for(int i = 0; i < src.rows; i += 5)
		for(int j = 0; j < src.cols; j+= 5)
			file << int(src.at<Vec3b>(i,j)[0]) << " "
			<< int(src.at<Vec3b>(i,j)[1]) << " " <<int(src.at<Vec3b>(i,j)[2]) << std::endl;
	file.close();
}
std::vector<ColorCorrector::TriangleInfo> ColorCorrector::InOutTriangle(std::vector<Vector3<float>> points, std::vector<unsigned long> indices)
{
	std::vector<ColorCorrector::TriangleInfo> triangles_vector;
	for(int i = 0; i < indices.size()/3; ++i)
	{
		ColorCorrector::TriangleInfo buffer;
		buffer.out_normal = VectorProduct(points[indices[i+1]] - points[indices[i]],points[indices[i+2]] - points[indices[i]]);
		for(int j = 0; j < points.size(); ++j)
		{
			if(ScalarProduct(buffer.out_normal,points[j] - points[indices[i]]) < 0)
				buffer.under.push_back(j);
			else
				buffer.upper.push_back(j);
		}
		triangles_vector.push_back(buffer);
	}
	return triangles_vector;
}
std::vector<unsigned long> ColorCorrector::GetGuts(const std::vector<TriangleInfo>& triangles)
{
	std::vector<unsigned long> guts_indices;
	for(auto& idx: triangles[0].under)
		for(auto& triangle : triangles)
			if(std::find(triangle.under.begin(), triangle.under.end(), idx) != triangle.under.end())
				guts_indices.push_back(idx);
	return guts_indices;
}
std::vector<std::pair<int,int>> ColorCorrector::GetCandidates(const Mat& dispersion)
{
	Mat copy;
	dispersion.convertTo(copy, CV_32FC1, 1.0, 0.0);
	std::vector<std::pair<int,int>> candidates;
	for(int y = 0; y < dispersion.rows; ++y)
		for(int x = 0; x < dispersion.cols; ++x)
				candidates.emplace_back(std::pair<int,int>(x,y));
	sort(candidates.begin(), candidates.end(), [dispersion](const std::pair<int,int> & a, const std::pair<int,int> & b) -> bool
		{
			return dispersion.at<float>(a.second,a.first) < dispersion.at<float>(b.second,b.first);
		});
	float eps = 0.1 * dispersion.at<float>(candidates[candidates.size()/2].second, candidates[candidates.size()/2].first);
	for(int i = 0; i < candidates.size(); ++i)
		if(dispersion.at<float>(candidates[i].second,candidates[i].first) > eps)
		{
			candidates.erase(candidates.begin() + i, candidates.end());
			break;
		}
	return candidates;
}
void ColorCorrector::AdaptiveRadius(std::vector<std::pair<int,int>>& candidates, const int& radius)
{
	for(auto&elem : candidates)
		for (int i = 0; i < candidates.size(); ++i)
			if (R(elem, candidates[i]) < radius && elem != candidates[i]) {
				candidates.erase(candidates.begin() + i);
				if (i != 0)
					--i;
			}
}
int ColorCorrector::CDistance(const Vec3b& a, const Vec3b& b)
{
	int distance = std::abs(a.val[0] - b.val[0]) + std::abs(a.val[1] - b.val[1]) + std::abs(a.val[2] - b.val[2]);
	bool ch = false;
	for(int i = 0; i < 3; ++i)
		if(std::abs(a.val[i] - b.val[i]) < distance/2)
			ch = true;
	return ch ? distance : 1000;
}
void ColorCorrector::ColorDistance(std::vector<std::pair<int, int>>& candidates, const Mat& img, const int side)
{
	for(auto&elem : candidates)
		for (int i = 0; i < candidates.size(); ++i)
			if (CDistance(img.at<Vec3b>(elem.second, elem.first), img.at<Vec3b>(candidates[i].second, candidates[i].first)) < side) {
				candidates.erase(candidates.begin() + i);
				if (i != 0)
					--i;
			}
}
std::vector<Vector3<float>> ColorCorrector::GetIntersectionPoints(const ConvexHullInfo& first, const ConvexHullInfo& second)
{
	std::vector<TriangleInfo> f_triangles = InOutTriangle(first.pointCloud, first.indices);
	std::vector<TriangleInfo> s_triangles = InOutTriangle(second.pointCloud, second.indices);
	std::vector<unsigned long> f_idx = GetGuts(f_triangles);
	std::vector<unsigned long> s_idx = GetGuts(s_triangles);
	std::vector<Vector3<float>> intersection_points;
	std::vector<unsigned long> ind_ = (f_idx.size() < s_idx.size()) ? f_idx : s_idx;
}
Mat ColorCorrector::sRGB2RGB(const Mat& src)
{
	cv::Mat copy(src.size(),CV_32FC3);
	src.convertTo(copy, CV_32FC3, 1.0/255.0, 0.0);
	for(int y = 0; y < src.rows; ++y)
		for(int x = 0; x < src.cols; ++x)
			for(int channel = 0; channel < 3; ++channel)
			{
				if(copy.at<Vec3f>(y,x)[channel] <= 0.04045)
					copy.at<Vec3f>(y,x)[channel] /= 12.92;
				else
					copy.at<Vec3f>(y,x)[channel] = float(pow((copy.at<Vec3f>(y,x)[channel] + 0.055)/1.055, 2.4));
			}
	copy.convertTo(copy, CV_8UC3, 255, 0.0);
	return copy;
}
Mat ColorCorrector::RGB2sRGB(const Mat& src)
{
	cv::Mat copy(src.size(),CV_32FC3);
	src.convertTo(copy, CV_32FC3, 1.0, 0.0);
	for(int y = 0; y < src.rows; ++y)
		for(int x = 0; x < src.cols; ++x)
			for(int channel = 0; channel < 3; ++channel)
			{
				if(copy.at<Vec3f>(y,x)[channel] <= 0.0031308)
					copy.at<Vec3f>(y,x)[channel] *= 12.92;
				else
					copy.at<Vec3f>(y,x)[channel] = float(pow(copy.at<Vec3f>(y,x)[channel],1.0/2.4) * 1.055 - 0.055);
			}
	return copy;
}
float ColorCorrector::Dispersion(Mat boost, Mat src, const int x, const int y, const int box_size)
{
	Rect for_boost(Point(y,x), Size(box_size,box_size));
	Mat B;
	boost(for_boost).copyTo(B);
	absdiff(B, src.at<float>(x,y),B);
	multiply(B,B,B);
	float sum = float(cv::sum(B)[0]);
	sum /= (box_size*box_size - 1);
	return sum;
	// Disp = avg_sum((B - A)**2)
}
Mat ColorCorrector::HeavyDispersion(Mat img, const int box_size) {
	const int extra_height = box_size / 2; // расширяем размерность исходного изображения,
	const int extra_width = box_size / 2;
	img.convertTo(img, CV_32FC1, 1.0/255.0, 0.0);
	Mat split_ed[3]; // src
	Mat boosted[3]; // boosted images
	Mat deviation[3];
	split(img, split_ed);
	for(auto &b : boosted)
		b = Mat(cv::Size(img.cols + 2 * extra_width, img.rows + 2 * extra_height), split_ed[0].type(), Scalar(0.5));
	Rect boost_rect(Point(0,0), Size(img.cols + 2 * extra_width, img.rows + 2 * extra_height));
	Rect src_rect(Point(0,0), Size(img.cols, img.rows));
	for(int i = 0; i < 3; ++i)
	{
		deviation[i] = Mat(img.size(), img.type(), Scalar(0));
		for(int k  = 0; k < img.rows; ++k)
			for(int l = 0; l < img.cols; ++l)
				boosted[i].at<float>(k + extra_width,l + extra_height) = split_ed[i].at<float>(k,l);
	}
	for(int c = 0; c < 3; ++c) {
		for (int i = 0; i < img.rows; ++i) {
			for (int j = 0; j < img.cols; ++j)
			{
				float res = Dispersion(boosted[c], split_ed[c], i, j, box_size);
				res = (res < 1) ? res : 1;
				deviation[c].at<float>(i, j) = res;
			}
		}
	}
	Mat result(Size(img.cols, img.rows), CV_32FC1, Scalar(0));
	for(int i = 0; i < deviation[0].rows; ++i)
		for(int j = 0; j < deviation[0].cols; ++j)
		{
			float res = (pow(deviation[0].at<float>(i, j) * deviation[1].at<float>(i, j) * deviation[2].at<float>(i, j),
				double(1.0 / 3.0)));
			res = (res < 1 ) ? res: 1;
			result.at<float>(i,j) = res;
		}
	GaussianBlur(result, result, Size(5,5), 0, 0);
	return result;
}
std::vector<Vector3<float>> ColorCorrector::GetBounds(const std::vector<std::pair<int,int>>& indexes, const Mat& img)
{
	//pair {x,y}->{cols, rows}->{rows,cols}
	// img given reshape 0.5 and BGR->RGB + sRGB->Linear RGB mode only CV_8UC3!
	std::vector<Vector3<float>> bounds;
	Vec3f buffer;
	for(auto& idx : indexes)
	{
		buffer = img.at<Vec3f>(idx.second,idx.first);
		bounds.emplace_back(float(buffer[0]), float(buffer[1]), float(buffer[2]));
	}
	return bounds;
}
Vector3<float> ColorCorrector::ProjectFirstPlane(const Vector3<float>& normal,const Vector3<float>& base_point, const Vector3<float>& point)
{
	float D = -ScalarProduct(normal, base_point);
	float t = -(ScalarProduct(normal,point) + D)/ ScalarProduct(normal,normal);
	return Vector3<float>{normal.x*t+point.x,normal.y*t+point.y,normal.z*t+point.z};
}
Vector3<float> ColorCorrector::ZeroFirstPlane(const Vector3<float>& normal,const Vector3<float>& base_point, const Vector3<float>& point)
{
	float D = -ScalarProduct(normal, base_point);
	float t = (point != Vector3<float>(0,0,0)) ? -(D/ ScalarProduct(normal,point)): -(D/ ScalarProduct(normal,normal));
	return point*t;
}
std::vector<Point2f> ColorCorrector::GetOnesProjection(const std::vector<Vector3<float>>& points)
{
	Vector3<float> buffer{1,1,1};
	Vector3<float> normal{1,1,1};
	Vector3<float> base_point{0,0,1};
	std::vector<Point2f> xy_pairs;
	for(auto& pt : points) {
		if(sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z) >= 0.1) {
			buffer = ZeroFirstPlane(normal, base_point, pt);
			buffer = GetGoodLook(buffer, -45, { -1, 1, 0 }, { 0, 0, 0 });
			xy_pairs.emplace_back(Point2f(buffer.x, buffer.y));
		}
	}
	return xy_pairs;
}
std::vector<Vector3<float>> ColorCorrector::GetAlphaMatrix(const float& angle, const Vector3<float>& bv)
{
	float r_angle = angle*3.1415/180;
	Vector3<float> l1{std::cos(r_angle) + (1-std::cos(r_angle))*bv.x*bv.x,
					  (1-std::cos(r_angle))*bv.x*bv.y - std::sin(r_angle)*bv.z,
					  (1-std::cos(r_angle))*bv.x*bv.z + std::sin(r_angle)*bv.y};
	Vector3<float> l2{(1-std::cos(r_angle))*bv.y*bv.x + std::sin(r_angle)*bv.z,
					  std::cos(r_angle) + (1-std::cos(r_angle))*bv.y*bv.y,
					  (1-std::cos(r_angle))*bv.y*bv.z - std::sin(r_angle)*bv.x};
	Vector3<float> l3{(1-std::cos(r_angle))*bv.z*bv.x - std::sin(r_angle)*bv.y,
					  (1-std::cos(r_angle))*bv.z*bv.y + std::sin(r_angle)*bv.x,
					  std::cos(r_angle) + (1-std::cos(r_angle))*bv.z*bv.z};
	return std::vector<Vector3<float>>{l1,l2,l3};
}
Vector3<float> ColorCorrector::GetGoodLook(const Vector3<float>& point, const float& angle, const Vector3<float>& bv, const Vector3<float>& shift)
{
	std::vector<Vector3<float>> vec = GetAlphaMatrix(angle,bv);
	return (Vector3<float>{ ScalarProduct(vec[0],point), ScalarProduct(vec[1],point), ScalarProduct(vec[2],point)}) + shift;
}
Mat ColorCorrector::Algorithm_ALPHA(Mat& src, const Mat& dst, const bool cov, const int wnd_for_cov, const int rd)
{
	Mat promo = src;
	Mat absolute_copy(promo);
	absolute_copy.convertTo(absolute_copy,CV_32FC3,1.0/255.0, 0.0);
	std::cout << "preparation...\n";
	Mat copy_src = Preparation(src);
	Mat copy_dst = Preparation(dst);
	std::cout << "dispersion...\n";
	Mat covar = HeavyDispersion(copy_src, 23); //return in CV_32FC3
	std::vector<std::pair<int,int>> candidates = GetCandidates(covar);
	std::cout << "adaptive radius...\n";
	AdaptiveRadius(candidates,20);
	copy_src.convertTo(copy_src, CV_32FC3, 1.0/255.0, 0.0);
	copy_dst.convertTo(copy_dst, CV_32FC3, 1.0/255.0, 0.0);// на всякий
	std::vector<Vector3<float>> unique_points = GetBounds(candidates, copy_src);
	std::vector<Vector3<float>> unique_points_processed = GetBounds(candidates, copy_dst);

	std::vector<std::vector<Vector3<float>>> data_in{unique_points,unique_points_processed};
	alpha_data &temp = (alpha_data &) data_in;
	nlopt_opt opt;
	double lb[3] = { 0.8, 0.8, 0.8};
	double ub[3] = { 1.1, 1.1, 1.1};/* lower bounds */
	opt = nlopt_create(NLOPT_LN_NELDERMEAD, 3);
	nlopt_set_min_objective(opt, alpha_func, &data_in );
	nlopt_set_lower_bounds(opt, lb);
	nlopt_set_upper_bounds(opt, ub);
	nlopt_set_min_objective(opt, alpha_func, &data_in );
	double minf;
	double alpha[3] = {1.05, 0.95, 1.05};
	std::cout << "calculating alpha...\n";
	nlopt_result result = nlopt_optimize(opt, alpha, &minf);

	std::cout << "{" <<alpha[0] << " " << alpha[1] << " " <<  alpha[2] << "}\n";
	std::cout << "applying alpha\n";
	for(int y = 0; y < absolute_copy.rows; ++y)
		for(int x = 0; x < absolute_copy.cols; ++x)
		{
			absolute_copy.at<Vec3f>(y, x)[0] *= alpha[0];
			absolute_copy.at<Vec3f>(y, x)[1] *= alpha[1];
			absolute_copy.at<Vec3f>(y, x)[2] *= alpha[2];
		}
	absolute_copy = RGB2sRGB(absolute_copy);
	return absolute_copy;
}
Mat ColorCorrector::Preparation(const Mat& src)
{
	Mat copy(src);
	copy = sRGB2RGB(copy);
	cv::resize(copy, copy, cv::Size(0,0), 0.5, 0.5);
	//imwrite("/home/anreydron/Desktop/for_prj/LinearRGB.jpeg", copy);
	return copy;
}
float ColorCorrector::CrossProd(const Vector3<float>& v3, const Vec3f& v3f)
{
	return v3.x*v3f[0] + v3.y*v3f[1] + v3.z*v3f[2];
}
std::vector<Vector3<float>> ColorCorrector::InverseMatrix(const std::vector<Vector3<float>>& vec)
{
	if(Det3x3(vec[0],vec[1],vec[2]) == 0)
		throw std::invalid_argument("Dividing by zero det\n");
	std::vector<Vector3<float>> inv;
	float det = Det3x3(vec[0],vec[1],vec[2]);
	Vector3<float> l1(Det2x2(vec[1].y,vec[2].y,vec[1].z,vec[2].z),
			 -Det2x2(vec[0].y,vec[2].y,vec[0].z,vec[2].z),
			  Det2x2(vec[0].y,vec[1].y,vec[0].z,vec[1].z));
	Vector3<float> l2(-Det2x2(vec[1].x,vec[2].x,vec[1].z,vec[2].z),
		Det2x2(vec[0].x,vec[2].x,vec[0].z,vec[2].z),
		-Det2x2(vec[0].x,vec[1].x,vec[0].z,vec[1].z));
	Vector3<float> l3(Det2x2(vec[1].x,vec[2].x,vec[1].y,vec[2].y),
		-Det2x2(vec[0].x,vec[2].x,vec[0].y,vec[2].y),
		Det2x2(vec[0].x,vec[1].x,vec[0].y,vec[1].y));
	inv.push_back(l1/det);
	inv.push_back(l2/det);
	inv.push_back(l3/det);
	return inv;
}
Mat ColorCorrector::Check2DHULL (const Mat& src, const Mat& dst, const bool cov, const int wnd_for_cov, const int rd)
{
	Mat absolute_copy_of_src = src;
	absolute_copy_of_src.convertTo(absolute_copy_of_src,CV_32FC3,1.0/255.0,0);
	std::cout << "preparation...\n";
	Mat copy_src = Preparation(src);
	Mat copy_dst = Preparation(dst);
	std::cout << "dispersion...\n";
	Mat covar = Dispersion(copy_src, Size(3,3)); //return in CV_32FC3
	std::vector<std::pair<int,int>> candidates = GetCandidates(covar);
	std::cout << "adaptive radius...\n";
	AdaptiveRadius(candidates,20);
	std::vector<std::pair<int,int>> cnds = candidates;
	ColorDistance(cnds,copy_src,40);
	copy_src.convertTo(copy_src, CV_32FC3, 1.0/255.0, 0.0);
	copy_dst.convertTo(copy_dst, CV_32FC3, 1.0/255.0, 0.0);

	std::vector<Vector3<float>> unique_points = GetBounds(cnds, copy_src);
	std::vector<Vector3<float>> unique_points_processed = GetBounds(cnds, copy_dst);


	std::vector<Point2f> coordinates_proj = GetOnesProjection(unique_points);
	std::vector<Point2f> coordinates_proj_processed = GetOnesProjection(unique_points_processed);
	std::vector<Point2f> hull;
	std::vector<Point2f> hull_dst;
	convexHull(coordinates_proj, hull);
	convexHull(coordinates_proj_processed, hull_dst);
	std::cout << "dst\n";
	std::cout << "[";
	for(auto &elem : hull_dst)
		std::cout << "[" << elem.x << "," << elem.y << "],\n";
	std::cout << "]\n";
	std::cout << "src\n";
	std::cout << "[";
	for(auto &elem : hull)
		std::cout << "[" << elem.x << "," << elem.y << "],\n";
	std::cout << "]\n";
	std::vector<std::vector<Point2f>> pf = Descriptor(hull, hull_dst);
	convex_data &temp = (convex_data &) pf;
	nlopt_opt opt;
	//double lb[6] = { -100, -100, -100, -100, -100, -100};
	//double ub[6] = { 100, 100, 100, 100, 100, 100};// lower bounds
	opt = nlopt_create(NLOPT_LN_NELDERMEAD, 6);
	nlopt_set_min_objective(opt, con2d_func, &pf );
	//nlopt_set_lower_bounds(opt, lb);
	//nlopt_set_upper_bounds(opt, ub);
	nlopt_set_min_objective(opt, con2d_func, &pf);
	double minf;
	double alpha[6] = {1, 0, 0, 1, 0, 0};
	std::cout << "calculating affine transformation...\n";
	nlopt_result result = nlopt_optimize(opt, alpha, &minf);
	std::cout << "[[" << alpha[0] << "," << alpha[1] << "]," << "[" << alpha[2] << "," << alpha[3] << "]]\n";
	std::cout << "[" << alpha[4] << "," << alpha[5] << "]\n";
	std::vector<Vector3<float>> forward = GetAlphaMatrix(-45,Vector3<float>{-1,1,0});
	std::vector<Vector3<float>> backward = InverseMatrix(forward);
	Vec3f buffer;
	for(int y = 0; y < absolute_copy_of_src.rows; ++y)
		for(int x = 0; x < absolute_copy_of_src.cols; ++x)
		{
			buffer = absolute_copy_of_src.at<Vec3f>(y,x);
			absolute_copy_of_src.at<Vec3f>(y,x)[0] = CrossProd(forward[0], buffer);
			absolute_copy_of_src.at<Vec3f>(y,x)[1] = CrossProd(forward[1], buffer);
			absolute_copy_of_src.at<Vec3f>(y,x)[2] = CrossProd(forward[2], buffer);
		}
	for(int y = 0; y < absolute_copy_of_src.rows; ++y)
		for(int x = 0; x < absolute_copy_of_src.cols; ++x)
		{
			buffer = absolute_copy_of_src.at<Vec3f>(y,x);
			absolute_copy_of_src.at<Vec3f>(y,x)[0] = buffer[0]*alpha[0] + buffer[1]*alpha[1] + alpha[4];
			absolute_copy_of_src.at<Vec3f>(y,x)[1] = buffer[0]*alpha[2] + buffer[1]*alpha[3] + alpha[5];
		}
	for(int y = 0; y < absolute_copy_of_src.rows; ++y)
		for(int x = 0; x < absolute_copy_of_src.cols; ++x)
		{
			buffer = absolute_copy_of_src.at<Vec3f>(y,x);
			absolute_copy_of_src.at<Vec3f>(y,x)[0] = CrossProd(backward[0], buffer);
			absolute_copy_of_src.at<Vec3f>(y,x)[1] = CrossProd(backward[1], buffer);
			absolute_copy_of_src.at<Vec3f>(y,x)[2] = CrossProd(backward[2], buffer);
		}
	/*cv::resize(copy_src, copy_src, cv::Size(0,0), 2, 2);
	copy_src = RGB2sRGB(copy_src);
	copy_src.convertTo(copy_src,CV_8UC3, 255.0,0);*/
	absolute_copy_of_src.convertTo(absolute_copy_of_src,CV_8UC3,255.0,0);
	imwrite("/home/anreydron/Desktop/for_prj/Results/Answer_zero.jpg", absolute_copy_of_src);
	return copy_src;
}
std::vector<std::vector<Point2f>> ColorCorrector::Descriptor(const std::vector<Point2f>& src, const std::vector<Point2f>& dst)
{
	std::vector<Point2f> target = (src.size() > dst.size()) ? dst : src;
	std::vector<Point2f> buf = (src.size() < dst.size()) ? dst : src;
	std::vector<std::vector<Point2f>> pairs(2);
	Point2f pt(1,1);
	for(auto& elem : target)
	{
		for (auto& el : buf)
		{
			Point2f  res = elem -el;
			res = elem - pt;
			if (norm((elem - el)) < norm((elem - pt)))
				pt = el;
		}
		pairs[0].emplace_back(elem);
		pairs[1].emplace_back(pt);
	}
	return pairs;
}