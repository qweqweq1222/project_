#include <iostream>
#include "libprj.h"
#include "nlopt.hpp"
#include <fstream>
void Poh(Mat src)
{
	Mat matrix(3,3, CV_32FC1,Scalar(0));
	Vec3f l1(3.2406,-1.5372,-0.4986);
	Vec3f l2(-0.9689,1.8758,0.0415);
	Vec3f l3(0.0557,-0.2040,1.0570);
	Mat xyz;
	cv::cvtColor(src, xyz,cv::COLOR_BGR2XYZ);
	xyz.convertTo(xyz, CV_32FC3, 1.0/255.0, 0);
	for(int i = 0; i < 10; ++i)
		for(int j = 0; j < 10; ++j)
			std::cout << xyz.at<Vec3f>(i,j) << std::endl;
	std::cout << xyz.size <<std::endl;
	for(int y = 0; y < src.rows - 10; ++y)
	{
		for (int x = 0; x < src.cols; ++x)
		{
			xyz.at<Vec3f>(y, x)[0] = float(xyz.at<Vec3f>(y, x).dot(l1));
			xyz.at<Vec3f>(y, x)[1] = float(xyz.at<Vec3f>(y, x).dot(l2));
			xyz.at<Vec3f>(y, x)[2] = float(xyz.at<Vec3f>(y, x).dot(l3));
		}
	}
	imshow("window", xyz);
	waitKey(0);
}
int main()
{

	auto flower = cv::imread("/home/anreydron/Desktop/for_prj/FlowerWithoutFocus.jpeg");
	auto flower_processed = cv::imread("/home/anreydron/Desktop/for_prj/FlowerWithoutFocus.jpg");
	ColorCorrector core(flower);
	std::vector<std::string> filenames = {"/home/anreydron/Desktop/for_prj/GaussDispersion/", "/home/anreydron/Desktop/for_prj/HeavyDispersion/", "/home/anreydron/Desktop/for_prj/LinearRGB/"};
	std::vector<Size> sizes = {Size(3,3), Size(7,7), Size(13,13),Size(23,23)};
	std::vector<std::string> strs = {"3.jpg", "7.jpg", "13.jpg", "23.jpg"};
	std::vector<std::string> strs_csv = {"3.csv", "7.csv", "13.csv", "23.csv"};
	Mat buffer;
	/*for(int i = 0; i < sizes.size(); ++i)
	{
		Mat copy = core.Preparation(flower);
		buffer = core.Preparation(flower);
		buffer = core.Dispersion(buffer, sizes[i]);
		std::ofstream out("/home/anreydron/Otladka/covar" + strs_csv[i]);
		for(int y = 0; y < buffer.rows; ++y)
			for(int x = 0; x < buffer.cols; ++x)
				out << buffer.at<float>(y,x) << "\n";
		out.close();
		//std::vector<std::pair<int,int>> candidates = core.GetCandidates(buffer);
		/*for(int y = 0; y < buffer.rows; ++y)
			for(int x = 0; x < buffer.cols; ++x)
			{
				float b = 1 - 2*buffer.at<float>(y,x);
				buffer.at<float>(y,x) = b > 0 ?  b : 0;
			}
		normalize(buffer,buffer,1,0,NORM_MINMAX);
		std::cout << buffer.at<float>(0,0);
		buffer.convertTo(buffer, CV_8UC1, 255.0, 0);
		imwrite(filenames[0] + strs[i], buffer);*/
		/*for(auto& pr : candidates)
			circle(copy,Point2i{pr.first,pr.second},1, Scalar(255,0,0));
		imwrite("/home/anreydron/Desktop/for_prj/CandidatesGaussDsp/" + strs[i], copy);


		buffer = core.Preparation(flower);
		buffer = core.HeavyDispersion(buffer, sizes[i].height);
		//candidates = core.GetCandidates(buffer);
		std::ofstream out_("/home/anreydron/Otladka/covar_heavy" + strs_csv[i]);
		for(int y = 0; y < buffer.rows; ++y)
			for(int x = 0; x < buffer.cols; ++x)
				out_ << buffer.at<float>(y,x) << "\n";
		out_.close();
		buffer.convertTo(buffer,CV_8UC1, 255.0, 0);
		copy = core.Preparation(flower);
		imwrite(filenames[1] + strs[i], buffer);
		for(auto& pr : candidates)
			circle(copy,Point2i{pr.first,pr.second},1, Scalar(255,0,0));
		imwrite("/home/anreydron/Desktop/for_prj/CandidatesHeavyDsp/" + strs[i], copy);
	}

	std::vector<std::string> names = {"/home/anreydron/Desktop/for_prj/AdaptiveRadiusGauss/33-",
									  "/home/anreydron/Desktop/for_prj/AdaptiveRadiusGauss/77-",
									  "/home/anreydron/Desktop/for_prj/AdaptiveRadiusGauss/1313-"};
	std::vector<std::string> names_color = {"/home/anreydron/Desktop/for_prj/AdaptiveRadiusColor/33-",
									  "/home/anreydron/Desktop/for_prj/AdaptiveRadiusColor/77-",
									  "/home/anreydron/Desktop/for_prj/AdaptiveRadiusColor/1313-"};

	for(int i = 0; i < 3; ++i)
	{
		Mat copy = core.Preparation(flower);
		buffer = core.Preparation(flower);
		buffer = core.Dispersion(buffer, sizes[i]);
		std::vector<std::pair<int, int>> candidates = core.GetCandidates(buffer);
		for (auto& rd : {20,50})
		{
			Mat bag = core.Preparation(flower);
			core.AdaptiveRadius(candidates, rd);
			std::cout << candidates.size() << std::endl;
			for (auto& idx : candidates)
			{
				line(bag,Point2i{ idx.first + 2, idx.second },Point2i{ idx.first + 7, idx.second },Scalar{ 155, 255, 0 });
				line(bag,Point2i{ idx.first - 2, idx.second },Point2i{ idx.first - 7, idx.second },Scalar{ 155, 255, 0 });
				line(bag,Point2i{ idx.first, idx.second + 2 },Point2i{ idx.first, idx.second + 7 },Scalar{ 155, 255, 0 });
				line(bag,Point2i{ idx.first, idx.second - 2 },Point2i{ idx.first, idx.second - 7 },Scalar{ 155, 255, 0 });
				circle(bag, Point2i{ idx.first, idx.second }, rd, Scalar{ 155, 255, 0 });
			}
			if(rd == 20)
				imwrite(names[i] + "20.jpg", bag);
			else
				imwrite(names[i] + "50.jpg", bag);
			bag = core.Preparation(flower);
			std::vector<std::pair<int, int>> cnds = candidates;
			core.ColorDistance(cnds, bag,40);
			for (auto& idx : cnds)
			{
				line(bag,Point2i{ idx.first + 2, idx.second },Point2i{ idx.first + 7, idx.second },Scalar{ 155, 255, 0 });
				line(bag,Point2i{ idx.first - 2, idx.second },Point2i{ idx.first - 7, idx.second },Scalar{ 155, 255, 0 });
				line(bag,Point2i{ idx.first, idx.second + 2 },Point2i{ idx.first, idx.second + 7 },Scalar{ 155, 255, 0 });
				line(bag,Point2i{ idx.first, idx.second - 2 },Point2i{ idx.first, idx.second - 7 },Scalar{ 155, 255, 0 });
				circle(bag, Point2i{ idx.first, idx.second }, rd, Scalar{ 155, 255, 0 });
			}
			if(rd == 20)
				imwrite(names_color[i] + "20.jpg", bag);
			else
				imwrite(names_color[i] + "50.jpg", bag);
		}
	}*/
		core.Check2DHULL(flower,flower_processed);
	return 0;
}
