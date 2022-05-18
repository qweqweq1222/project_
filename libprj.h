//
// Created by anreydron on 20.04.2022.
//
#ifndef PRJ__LIBPRJ_H
#define PRJ__LIBPRJ_H
#include <iostream>
#include <vector>
#include <cmath>
#include <stack>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgcodecs.hpp"
#include <iomanip>
#include <fstream>
#include <algorithm>
#include "nlopt.hpp"
#include "quickhull/QuickHull.hpp"
using namespace cv;
using namespace quickhull;
using namespace nlopt;
class ColorCorrector
{
	struct ConvexHullInfo
	{
		std::vector<Vector3<float>> pointCloud;
		ConvexHull<float> mesh;
		std::vector<unsigned long> indices;
	};
	struct TriangleInfo
	{
		Vector3<float> out_normal;
		std::vector<unsigned long> under;
		std::vector<unsigned long> upper;
	};
public:
	ColorCorrector() = default; // done
	~ColorCorrector() = default; // done
	ColorCorrector(Mat src_img):src(src_img){} // done
	Mat Dispersion(const Mat& src_img, const Size& size); //done
	static std::vector<std::pair<int,int>> GetCandidates(const Mat& dispersion); // done
	void DropDotsToFile(const std::string& str_xy, const Mat& src); //done
	std::vector<Vector3<float>> GetIntersectionPoints(const ConvexHullInfo& first, const ConvexHullInfo& second); //todo
	void AdaptiveRadius(std::vector<std::pair<int,int>>& candidates, const int& radius); // done
	static Mat sRGB2RGB(const Mat& src); //done
	Mat RGB2sRGB(const Mat& src);
	static Mat HeavyDispersion(Mat img, const int box_size); //done
	std::vector<Vector3<float>> GetBounds(const std::vector<std::pair<int,int>>& indexes, const Mat& img); //done
	std::vector<Point2f> GetOnesProjection(const std::vector<Vector3<float>>& points);
	ConvexHullInfo CHull(const std::vector<Vector3<float>>& points); //done
	Mat Algorithm_ALPHA(Mat& src, const Mat& destination,  const bool heavy_dispersion = true, const int window_for_dispersion = 23, const int radius = 20);
	Mat Preparation(const Mat& src);
	Mat Check2DHULL (const Mat& src, const Mat& destination, const bool heavy_dispersion = true, const int window_for_dispersion = 23, const int radius = 20); // todo преобразование переделать
	void ColorDistance(std::vector<std::pair<int,int>>& candidates, const Mat& img, const int side);
	std::vector<Vector3<float>> InverseMatrix(const std::vector<Vector3<float>>& vec);
	Vector3<float> ZeroFirstPlane(const Vector3<float>& normal,const Vector3<float>& base_point, const Vector3<float>& point);
private:
	Mat src;
	Vector3<float> VectorProduct(const Vector3<float>& lh, const Vector3<float>& rh); //done
	static float ScalarProduct(const Vector3<float>& lh, const Vector3<float>& rh); //done
	float Volume(const std::vector<unsigned long>& indices, const std::vector<Vector3<float>>& pointCloud); //done
	float TetraVolume(const Vector3<float>& r1,const Vector3<float>& r2,const Vector3<float>& r3,const Vector3<float>& r4); //done
	static float Det2x2(const float& l1l, const float& l1r, const float& l2l, const float& l2r); //done
	float Det3x3(const Vector3<float>& l1, const Vector3<float>& l2, const Vector3<float>& l3); //done
	float R(std::pair<int,int>& lf , std::pair<int,int>& rh); //done
	std::vector<TriangleInfo> InOutTriangle(std::vector<Vector3<float>> points, std::vector<unsigned long> indices); //done
	static std::vector<unsigned long> GetGuts(const std::vector<TriangleInfo>& triangles); //done
	static float Dispersion(Mat boost, Mat src, const int x, const int y, const int box_size); //done
	static Vector3<float> ProjectFirstPlane(const Vector3<float>& normal, const Vector3<float>& base_point, const Vector3<float>& point);//done
	Vector3<float> GetGoodLook(const Vector3<float>& point, const float& angle, const Vector3<float>& based_vector, const Vector3<float>& shift); //done
	std::vector<std::vector<Point2f>> Descriptor(const std::vector<Point2f>& src, const std::vector<Point2f>& dst);
	std::vector<Vector3<float>> GetAlphaMatrix(const float& angle, const Vector3<float>& bv);
	int CDistance(const Vec3b& a, const Vec3b& b);
	float CrossProd(const Vector3<float>& v3, const Vec3f& v3f);
	typedef struct {
		std::vector<std::vector<Vector3<float>>> data;
	} *alpha_data;
	typedef struct {
		std::vector<std::vector<Point2f>> data;
	} *convex_data;
	static double alpha_func(unsigned n, const double *x, double *grad, void* mfdoom)
	{
		float loss;
		alpha_data& data = (alpha_data &) mfdoom;
		std::vector<std::vector<Vector3<float>>> vec = data->data;
		for(int i = 0; i < data->data[0].size(); ++i)
		{
			loss += std::pow(vec[0][i].x*x[0] - vec[1][i].x,2);
			loss += std::pow(vec[0][i].y*x[1] - vec[1][i].y,2);
			loss += std::pow(vec[0][i].z*x[2] - vec[1][i].z,2);
		}
		return loss;
	}
	static double con2d_func(unsigned n, const double *x, double *grad, void* mfdoom)
	{
		// [a b c d]  + [e f]
		float loss;
		convex_data& data = (convex_data &) mfdoom;
		std::vector<std::vector<Point2f>> vec = data->data;
		for(int i = 0; i < data->data[0].size(); ++i)
		{
			loss += std::pow(x[0] * vec[1][i].x + x[1] *vec[1][i].y + x[4] - vec[0][i].x,2);
			loss += std::pow(x[2] * vec[1][i].x + x[3] *vec[1][i].y + x[5] - vec[0][i].y,2);
		}
		return loss;
	}
};
#endif //PRJ__LIBPRJ_H
